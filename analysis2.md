# Vertex AI endpoint observability without console access

**Status:** Analysis complete. Recommendation updated for the existing GCP OTel Collector, Elasticsearch and Kibana; pending confirmation from that stack's owners and the on-prem spike for B.
**Date:** 2026-10-23
**Scope:** Logs and operational metrics for Vertex AI online prediction endpoints (`mle-project-496007`, `us-central1`), delivered to users who have no GCP console access.
**Related:** [Logging-cost levers](2026-09-09-vertex-logging-cost.md) · [Exclusion + GCS sink runbook](logs-exclusion-gcs-sink.md) · [Endpoint access control](superpowers/specs/2026-09-24-endpoint-access-control-design.md)

---

## 1. Summary

Today, the only way to see endpoint logs and metrics is the GCP console under *Vertex AI → Endpoints*. Users, including engineers on consuming teams, will not get console access. We need another way to retrieve and render this data that:
- fits a **one-way network** (on-prem → GCP only);
- reuses what already exists: the **on-prem Grafana, Prometheus and Thanos stack**, and the **OTel Collector, Elasticsearch and Kibana already running in GCP**.

**Recommendation:**

| Signal | Approach | Why |
|---|---|---|
| **Metrics** | **Option B:** on-prem Grafana queries Cloud Monitoring with PromQL | Near-zero build, pull-only, built-in metrics are free to ingest. Proven locally |
| **Logs** | **Option E-hybrid on the existing stack:** Log Router sink → Pub/Sub → **existing OTel Collector** → **existing Elasticsearch**. Search in the **existing Kibana**, scoped per team, with optional log panels in the on-prem Grafana | No new platform, logs never leave GCP, Pub/Sub buffers outages, every container log line is captured, no predictor code change, no 60 req/min quota |
| **Custom model metrics** | **Defer.** When needed: OTel SDK → OTLP → Cloud Monitoring (visible in B), or to the existing collector if Kibana is the main UI | Sample-based billing is ~25× cheaper for histograms than byte-based |
| **Portal** | Deep links or embeds of Kibana and Grafana. A native portal log view can query Elasticsearch with per-team API keys | No quota-bound `entries.list`; no rebuilding of charts |

- **Fallback for logs: option D** (Pub/Sub → Alloy → on-prem Loki), if the existing Elasticsearch can't take the volume or tenancy, or if logs must be searchable from the on-prem stack only.
- **Not recommended: variant E-direct** (predictor sends OTLP straight to the collector, no Pub/Sub). It loses crash, OOM and startup logs, has no outage buffer, and needs code changes in every predictor.

```mermaid
flowchart LR
    subgraph gcp["GCP"]
        V["Vertex AI endpoint"] -->|"built-in metrics (free)"| CM[("Cloud Monitoring")]
        V -->|"stdout / stderr"| LR["Log Router"] -->|"Pub/Sub sink"| PS[["Pub/Sub<br/>pull subscription"]]
        LR -->|"GCS sink"| GCS[("GCS archive")]
        PS -->|"googlecloudpubsub receiver"| OC["Existing OTel Collector"]
        OC -->|"elasticsearch exporter<br/>per-team data streams"| ES[("Existing Elasticsearch")]
        ES --- KB["Existing Kibana<br/>Space per team"]
    end
    subgraph onprem["On-prem"]
        G["Grafana"]
        U["Users"]
    end
    G -->|"B: PromQL, outbound"| CM
    U -->|"browser, outbound HTTPS"| KB
    G -.->|"optional: Elasticsearch datasource"| ES
```

**The single most important lever isn't a tool choice:** cut log volume at the source (log level, no payload logging, no health-probe lines). It improves cost, quota headroom, PII exposure, Elasticsearch storage and search quality at the same time.

**Indicative incremental running cost** (list prices, a 20-endpoint fleet, 100 GiB of logs per month after source reduction): **≈ $35–$85/month** on GCP services, dominated by dashboard reads and `_Default` storage (Pub/Sub ≈ $7, GCS ≈ $2), **plus the share of the existing Elasticsearch cluster's storage** (unverified; set by retention). There's no GCP → on-prem egress for logs. Details are in §6.

---

## 2. Problem and constraints

### 2.1 Requirements

| # | Requirement |
|---|---|
| R1 | Users without GCP console access can see endpoint health (traffic, latency, errors, utilisation, replicas) |
| R2 | Users can search the prediction container's logs for their own models |
| R3 | Access is scoped per team or model; logs can contain PII (`race`, `sex`, `marital_status`, … in the census payload) |
| R4 | Works over a **one-way** network: on-prem can reach Google APIs through a per-destination firewall whitelist; GCP cannot initiate connections to on-prem |
| R5 | Reuses existing platforms where possible: on-prem **Grafana + Prometheus + Thanos**, and the GCP-hosted **OTel Collector + Elasticsearch + Kibana** |
| R6 | Predictable cost, including at high cardinality and with many viewers |

### 2.2 The two APIs everything sits on

- **Cloud Logging API v2** ([overview](https://docs.cloud.google.com/logging/docs/reference/api-overview)):
  - `entries.list` reads log buckets.
  - Google states it is *"not intended for high-volume retrieval"* and recommends sinks for bulk.
  - **Quota: 60 requests/min per project, not increasable.**
- **Cloud Monitoring API v3** ([overview](https://docs.cloud.google.com/monitoring/api/v3)):
  - `timeSeries.list` (filters plus server-side aggregation).
  - A **PromQL** endpoint (`projects.location.prometheus.api.v1`).
  - `timeSeries.create` for custom metrics.
  - MQL (`timeSeries.query`) lost Google customer support on 2025-07-22, so don't build on it.
  - Metric retention: 24 months (raw for 6 weeks, then 10-minute downsampling).

Because of R4, every viable design has **on-prem pull from Google APIs**. Prometheus `remote_write`, OTLP push, Thanos Receive and Pub/Sub *push* subscriptions are ruled out.

---

## 3. What we need to observe

### 3.1 Metrics: three cost categories

A metric's cost category is set by **its prefix and how it was sent**, not by what it measures ([pricing](https://cloud.google.com/stackdriver/pricing), checked 2026-10-23).

| | **Built-in** | **Custom, byte-based** | **Custom, sample-based** |
|---|---|---|---|
| Prefixes | `aiplatform.googleapis.com/prediction/online/*` | `custom.`, `workload.`, `external.googleapis.com/*`, user-defined log-based metrics | `prometheus.googleapis.com/*` (OTLP, Managed Prometheus) |
| Emitted by | Vertex AI, automatically | Our predictor | Our predictor |
| Answers | *Is the endpoint healthy?* | *Is the model behaving?* | *Is the model behaving?* |
| Examples | `prediction_count`, `response_count`, `error_count`, `prediction_latencies` (distribution → p50/p95/p99), `cpu/utilization`, `memory/bytes_used`, `accelerator/*`, `replicas` | Prediction-class mix, confidence histogram, feature null rate | Same |
| Labels | Fixed by Google (`endpoint_id`, `deployed_model_id`, `response_code`, …) | Ours: ≤30 per metric, ≤200k active series per resource | Ours |
| Metered as | Not metered | 8 B per scalar point, 80 B per distribution point | 1 sample per scalar point; distribution = 2 + 1 per non-zero bucket |
| Ingest price | **Free** | $0.258/MiB (tiered down to $0.061) | $0.06 per M samples (tiered down to $0.024) |
| Free allotment | n/a | 150 MiB per billing account per month | None |

The built-in metrics fully satisfy **R1**. Custom metrics are only needed for model-quality questions, which aren't in scope today (§9).

### 3.2 Logs

- The prediction container's stdout/stderr is forwarded to Cloud Logging under `resource.type="aiplatform.googleapis.com/Endpoint"`.
- Volume = verbosity × health-probe frequency × replicas + per-request lines.
- The earlier logging-cost work found the container writes **two lines per health probe, forever, even at zero traffic**, plus full request payloads at INFO. That makes it both a cost problem and a PII problem.

---

## 4. Options

```mermaid
flowchart LR
    subgraph gcp["GCP project"]
        V["Vertex AI endpoint"]
        CM[("Cloud Monitoring")]
        CL[("Cloud Logging<br/>Log Router")]
        PS[["Pub/Sub"]]
        P["Portal (Cloud Run)"]
        V --> CM
        V --> CL
        CL -- "sink" --> PS
        P -- "A: API reads" --> CM & CL
    end
    subgraph onprem["On-prem (outbound only)"]
        U["Users"]
        G["Grafana"]
        EX["stackdriver_exporter"] --> PR["Prometheus"] --> TH[("Thanos")] --> G
        AL["Alloy"] --> LK[("Loki")] --> G
    end
    FW{{"Firewall<br/>outbound → *.googleapis.com"}}
    U -- "A" --> FW --> P
    G -- "B" --> FW --> CM
    EX -- "C" --> FW --> CM
    AL -- "D" --> FW --> PS
```

| Option | Signal | Summary | Build effort |
|---|---|---|---|
| **A** Portal UI | Metrics + logs | The portal calls the Monitoring/Logging APIs and renders its own charts and log table | High |
| **B** Grafana → Cloud Monitoring | Metrics | On-prem Grafana queries Cloud Monitoring live with PromQL | **Low** |
| **C** Exporter → Thanos | Metrics | An on-prem exporter polls Cloud Monitoring into Prometheus/Thanos | Medium |
| **D** Sink → Pub/Sub → Loki | Logs | A Log Router sink streams logs to Pub/Sub; on-prem Alloy pulls them into Loki | Medium |
| **E** GCP stack (**existing** collector, Elasticsearch, Kibana) | Logs (+ metrics) | Logs: Pub/Sub → existing collector → existing Elasticsearch, searched in Kibana. Metrics stay on B (E-hybrid) or go to Elasticsearch. Variant E-direct skips Pub/Sub | **Low–Medium** (hybrid on existing stack) |

### 4.1 Option A: portal UI on the Logging and Monitoring APIs

The portal (in GCP) calls `timeSeries.list` (aggregated) and `entries.list`, maps the user's ADA SSO group to permitted `endpoint_id`s, and renders the results.

```mermaid
flowchart LR
    U["User (on-prem)"] -->|"ADA SSO"| FW{{"Firewall"}} --> P["Portal"]
    P -->|"timeSeries.list, aggregated"| CM[("Cloud Monitoring")]
    P -->|"entries.list, 60 req/min/project"| CL[("Cloud Logging")]
    P --- C[("Cache")]
```

| Pros | Cons |
|---|---|
| One URL and one auth model; users need no GCP identity | Rebuilds Grafana: charts, time pickers, log search, pagination |
| Fine-grained per-model authorization, which matters for PII | **Logs are capped by the 60 req/min `entries.list` quota**, shared project-wide; needs caching and rate limits |
| Full control of aggregation, so read cost is bounded by design | No alerting, ad-hoc exploration or correlation with on-prem dashboards |
| | Only sees logs that are **stored in a log bucket**, so it breaks if `_Default` excludes them (§7) |

### 4.2 Option B: on-prem Grafana queries Cloud Monitoring (recommended for metrics)

Add Grafana's built-in **Google Cloud Monitoring** datasource (it appears as `stackdriver` in the datasource picker) to the existing on-prem Grafana. Query it with the **PromQL** query type. The data stays in GCP; Grafana reads it on each refresh.

```mermaid
flowchart LR
    U["Users"] --> G["On-prem Grafana<br/>Google Cloud Monitoring datasource<br/>query type: PromQL"]
    G -->|"HTTPS, outbound<br/>monitoring.viewer only"| FW{{"Firewall"}} --> CM[("Cloud Monitoring")]
```

**How:**
1. Create a service account with `roles/monitoring.viewer` only. Use Workload Identity Federation if available; otherwise a key in Grafana's secret store.
2. Whitelist `monitoring.googleapis.com` and `oauth2.googleapis.com`, or the `private.googleapis.com` /30.
3. Build dashboards with one PromQL expression per query, always aggregated. Set the minimum interval to 1m or more and the refresh to 1m or more.
4. Create one Grafana folder per team, with an `endpoint_id` variable.

| Pros | Cons |
|---|---|
| Reuses existing dashboards, alerting and PromQL skills. Days, not weeks | One service account sees all project **metrics**. Scoping is by Grafana folder. Acceptable because metrics carry no PII |
| Pull-only, one firewall entry, no new infrastructure | **Every refresh is a billable read** ($0.50 per M series). Unaggregated dashboards get expensive (§6.2) |
| Built-in metrics are free to ingest; custom OTLP metrics appear automatically | History is GCP's (24 months, downsampled after 6 weeks), not Thanos |

Don't use Grafana's Cloud Logging datasource the same way: it would give Grafana project-wide access to PII-bearing logs. Use D for logs.

### 4.3 Option C: exporter → on-prem Prometheus/Thanos

An exporter **running on-prem** (`prometheus-community/stackdriver_exporter`) polls Cloud Monitoring on a fixed schedule. Prometheus scrapes it into Thanos.

```mermaid
flowchart LR
    subgraph onprem["On-prem"]
        EX["stackdriver_exporter<br/>prefix allow-list"] --> PR["Prometheus"] --> TH[("Thanos")] --> G["Grafana"]
    end
    EX -->|"poll every 5 min, outbound"| FW{{"Firewall"}} --> CM[("Cloud Monitoring")]
```

**How:**
- Use the same service account and firewall entry as B.
- Exporter settings:
  - `--monitoring.metrics-type-prefixes=aiplatform.googleapis.com/prediction/online`, adding `prometheus.googleapis.com/<custom prefix>` once custom metrics exist.
  - `--monitoring.metrics-interval=5m`
  - `--monitoring.metrics-offset=3m`
  - The project flag (`--google.project-ids` or `--google.project-id`) depends on the release.
- Add recording rules for p95 latency and error rate per endpoint.

| Pros | Cons |
|---|---|
| One long-term store with on-prem retention, recording rules and correlation | We own the exporter: config, prefix allow-list, lag tuning |
| Read cost is fixed by the poll schedule, **independent of viewer count** | Polling lag of 5–8 minutes on top of GCP's ingest delay |
| Also carries custom `prometheus.googleapis.com` metrics, so a GCP-side collector is rarely needed | Cardinality moves into Thanos unless the exporter is filtered |
| | Metrics only; logs still need D |

A variant (C2) puts an OTel Collector in GCP behind a public authenticated `/metrics` endpoint, scraped by on-prem Prometheus (outbound from on-prem). It's only worth it if custom metrics must bypass Cloud Monitoring entirely, which no current requirement needs.

### 4.4 Option D: Log Router sink → Pub/Sub → on-prem Loki (fallback for logs)

A **Log Router sink** streams matching Vertex log entries into a Pub/Sub topic. On-prem **Grafana Alloy** pulls from a subscription (outbound only) and writes to Loki. Users search logs in the existing Grafana, scoped by Loki tenant.

```mermaid
flowchart LR
    subgraph gcp["GCP"]
        V["Vertex endpoint"] -->|"stdout / stderr"| LR["Log Router"]
        LR -->|"_Default sink<br/>narrow exclusion"| DB[("_Default bucket")]
        LR -->|"Pub/Sub sink"| T[["Topic<br/>storage policy: us-central1"]]
        LR -->|"GCS sink"| GCS[("GCS archive")]
        T --> S1[["Pull subscription<br/>7-day retention"]]
    end
    subgraph onprem["On-prem"]
        AL["Alloy<br/>loki.source.gcplog (pull)"] --> LK[("Loki<br/>tenant per team")] --> G["Grafana"]
    end
    AL -->|"outbound gRPC<br/>subscriber-only SA"| FW{{"Firewall"}} --> S1
```

| Pros | Cons |
|---|---|
| **No `entries.list` quota involved**; near-real-time | New components to run: sink, topic, Alloy, Loki capacity |
| PII stays scoped per team in Loki; Grafana never gets project-wide log access | Loki storage, retention and limits become our responsibility |
| Cheap: Pub/Sub ≈ $7 per 100 GiB (§6.3) | Delivers only entries routed after the sink exists, so no backfill from Pub/Sub |
| Buffered: an on-prem outage loses nothing within the 7-day subscription retention | Logs leave GCP, so they need egress capacity and a residency decision (§8) |

Implementation steps and commands are in [Appendix A](#appendix-a--option-d-setup).

### 4.5 Option E: observability stack in GCP (OTel Collector + ELK + Grafana)

> **Update: the OTel Collector, Elasticsearch and Kibana already exist in GCP.** E's main drawbacks below (*"a new platform to operate"*, *"Elasticsearch cost dominates"*) mostly fall away: the platform, its owners and its cost base exist already. The remaining work is the sink, Pub/Sub, a collector pipeline and per-team data streams. The recommended shape is **E-hybrid with the on-prem Grafana**: logs flow through Pub/Sub → existing collector → existing Elasticsearch, users search in Kibana, and metrics stay on B. A GCP-hosted Grafana is optional, not required.

The **whole stack runs in GCP**: an OpenTelemetry Collector, Elasticsearch, Kibana (plus Logstash if wanted), and a **GCP-hosted Grafana** that reads Elasticsearch through its Elasticsearch datasource. Nothing is installed on-prem. Users open Grafana or Kibana in their browser, which is an ordinary outbound HTTPS request from on-prem, so R4 holds without any data leaving GCP.

```mermaid
flowchart LR
    subgraph gcp["GCP project (us-central1)"]
        V["Vertex endpoint"] -->|"stdout / stderr"| LR["Log Router"] -->|"Pub/Sub sink"| PS[["Pull subscription"]]
        V -->|"built-in metrics"| CM[("Cloud Monitoring")]
        V -.->|"custom metrics, OTLP"| OC
        PS -->|"googlecloudpubsub receiver"| OC["OTel Collector<br/>GKE / GCE"]
        CM -->|"googlecloudmonitoring receiver<br/>(or read directly by Grafana)"| OC
        OC -->|"elasticsearch exporter<br/>per-team data streams"| ES[("Elasticsearch")]
        ES --- KB["Kibana"]
        ES --> GF["Grafana<br/>Elasticsearch datasource"]
        CM -.->|"optional: Cloud Monitoring datasource<br/>(Workload Identity)"| GF
        LB["HTTPS load balancer<br/>Cloud Armor: corporate egress IPs<br/>ADA SSO (OIDC/SAML)"] --> GF & KB
    end
    U["Users (on-prem browser)"] -->|"outbound HTTPS"| FW{{"Firewall"}} --> LB
```

**What changes because everything is in GCP:**
- **No GCP → on-prem data flow and no bulk egress.** Logs and metrics stay in the GCP region. Only rendered pages reach browsers, which removes D's egress, residency and Loki-capacity questions.
- **No key files.** The collector and Grafana use **Workload Identity / attached service accounts**: Pub/Sub subscriber and Monitoring viewer for the collector, Monitoring viewer for Grafana.
- **Only one firewall entry:** the user-facing Grafana/Kibana hostname. It replaces the per-API entries for `monitoring`, `pubsub` and `oauth2.googleapis.com`.
- **Metrics don't have to go through Elasticsearch.** A GCP-hosted Grafana can query Cloud Monitoring directly with the PromQL datasource, which is option B relocated into GCP. That keeps PromQL for metrics and uses Elasticsearch for logs only. This hybrid, **E-hybrid**, avoids Elasticsearch's weakest area (metrics) and is the recommended shape of E.
- **It's a new, separate stack.** It doesn't reuse the on-prem Grafana, Prometheus or Thanos, so R5 is ✕. Users get a second Grafana with its own folders, dashboards and alerting.

**Collection layer choices:**

| Signal | OTel Collector (default) | Elastic-native alternative |
|---|---|---|
| Logs | `googlecloudpubsub` receiver + `googlecloudlogentry` encoding extension (receiver beta; the built-in `cloud_logging` encoding was removed in v0.132.0) → `elasticsearch` exporter | Logstash `google_pubsub` input, or the Elastic Agent GCP integration (unverified here) |
| Built-in metrics | **E-hybrid:** none, because Grafana reads Cloud Monitoring directly. **Full E:** `googlecloudmonitoring` receiver (alpha; min 60s, default 300s) → Elasticsearch | Metricbeat / Elastic Agent GCP metrics (unverified here) |
| Custom metrics | Predictor OTel SDK → OTLP → collector `otlp` receiver → Elasticsearch. **E-hybrid:** OTLP → Cloud Monitoring (§9) instead, so they appear next to the built-in metrics | n/a |

Use one collection technology, not both. The collector is the vendor-neutral default; Logstash or Agent only make sense if an Elastic team already standardises on them.

**Implementation notes:**
- **Hosting:**
  - Run the collector on GKE or GCE, with 2+ replicas on the shared subscription. In full E, run the Monitoring receiver on **one** replica only.
  - Elasticsearch options: Elastic Cloud on GCP in `us-central1`, or self-managed with ECK on GKE.
  - Grafana runs on GKE or Cloud Run with a Cloud SQL backend.
- **User access path:**
  - Put Grafana and Kibana behind a regional HTTPS load balancer with a managed certificate.
  - Use Cloud Armor to allow only corporate egress IP ranges.
  - Log users in with **ADA SSO** through Grafana's generic OAuth/SAML. Kibana SSO (SAML/OIDC realms) may need a paid Elastic subscription (unverified).
  - IAP is an option only with Workforce Identity Federation, since users have no Google identities.
- **Per-team scoping:**
  - Use a `transform` processor that sets `data_stream.namespace` from an `endpoint_id` → team map. The exporter writes to `logs-vertex-<team>`.
  - Grafana uses one Elasticsearch datasource per team, each with an API key restricted to that team's indices, plus Grafana team/folder permissions.
  - Index-level privileges are enough; document-level security isn't required.
- **Versions:** the exporter's default `otel` mapping needs Elasticsearch ≥ 8.12 (≥ 8.16 recommended).
- **Delivery:** messages are acked only after export, so Pub/Sub's 7-day retention bounds a collector or Elasticsearch outage.

Illustrative collector config for E-hybrid, logs only. Field names are checked against the contrib READMEs; the values need adapting:

```yaml
extensions:
  googlecloudlogentry_encoding: {}

receivers:
  googlecloudpubsub:
    subscription: projects/mle-project-496007/subscriptions/vertex-endpoint-logs-collector
    encoding: googlecloudlogentry_encoding
  # full E only:
  # googlecloudmonitoring:
  #   project_id: mle-project-496007
  #   collection_interval: 60s
  #   metrics_list:
  #     - metric_descriptor_filter: 'metric.type = starts_with("aiplatform.googleapis.com/prediction/online/")'

processors:
  memory_limiter: { check_interval: 1s, limit_percentage: 80 }
  transform/team:            # set data_stream.namespace from an endpoint_id → team mapping
    log_statements: []       # fill in

exporters:
  elasticsearch:
    endpoint: https://<es-endpoint>:443
    api_key: ${env:ES_API_KEY}

service:
  extensions: [googlecloudlogentry_encoding]
  pipelines:
    logs: { receivers: [googlecloudpubsub], processors: [memory_limiter, transform/team], exporters: [elasticsearch] }
```

| Pros | Cons |
|---|---|
| **Fully inside GCP:** no on-prem install, no data egress to on-prem, residency in one region; PII never leaves GCP | **A new platform to operate:** Elasticsearch, Kibana and Grafana need HA, upgrades, backups, capacity and on-call. Duplicates tools that already exist on-prem (R5 ✕) |
| **Keyless:** Workload Identity for every component; one user-facing firewall entry | **Elasticsearch cost dominates** (unverified), plus Grafana and collector compute, load balancer and Cloud SQL |
| Strong full-text log search in Kibana or Grafana; per-team indices give clean R3 scoping | Users get a second Grafana, separate from the on-prem one: no shared dashboards, alerting or correlation with on-prem services |
| E-hybrid keeps PromQL on Cloud Monitoring for metrics, so Elasticsearch only holds logs | We own the user access path: load balancer, certificate, Cloud Armor and SSO integration, which is a security review item. Kibana SSO may need a paid licence |
| Full `_Default` exclusion (stage 4) without losing searchable logs; Logs Explorer is no longer needed | Collector receivers are alpha/beta, so pin versions. Full E puts metrics in Elasticsearch: no PromQL, less efficient storage |

**Cost** at the same scale as §6 (100 GiB/month of logs, 20 endpoints):
- Pub/Sub: ≈ $7.4.
- Metric reads:
  - E-hybrid: Grafana reads Cloud Monitoring as in §6.2, about $25/month for aggregated dashboards, more if unaggregated.
  - Full E: fixed polling, about $2.
- No GCP → on-prem egress.
- Unverified and dominant: Elasticsearch (daily ingest × index overhead × (1 + replicas) × retention days), Grafana and collector compute, the load balancer and Cloud SQL.

**When E beats B + D:** the organisation wants observability for GCP workloads to live **in GCP**, with no on-prem dependency, data egress or on-prem capacity ask. There's a team willing to own Elasticsearch and Grafana in GCP, and users accept a separate Grafana. If reusing the on-prem stack (R5) matters more, B + D remains the better fit.

#### Variant E-direct: collector, Elasticsearch and Kibana in GCP; Grafana on-prem; no Pub/Sub

Everything flows **through Elasticsearch**. There's no Log Router sink and no Pub/Sub:
- The predictor sends **logs and custom metrics over OTLP** straight to the collector.
- The collector polls **built-in metrics** from Cloud Monitoring.
- The **existing on-prem Grafana** reads Elasticsearch through its Elasticsearch datasource. Kibana in GCP is optional.

```mermaid
flowchart LR
    subgraph gcp["GCP project (us-central1)"]
        subgraph vtx["Vertex AI endpoint"]
            PR["Predictor container<br/>OTel SDK: logs + custom metrics"]
        end
        PR -->|"OTLP/HTTP :4318 (http/protobuf)<br/>internal LB / authenticated URL"| OC["OTel Collector<br/>otlp receiver, HTTP :4318 only"]
        VB["Vertex built-in metrics"] --> CM[("Cloud Monitoring")]
        CM -->|"googlecloudmonitoring receiver"| OC
        OC -->|"elasticsearch exporter"| ES[("Elasticsearch")]
        ES --- KB["Kibana"]
        ESLB["HTTPS endpoint<br/>Egress IPs"] --> ES
    end
    subgraph onprem["On-prem"]
        G["Existing Grafana<br/>Elasticsearch datasource<br/>"]
        U["Users"] --> G
    end
    G -->|"outbound HTTPS<br/>one firewall entry"| FW{{"Firewall"}} --> ESLB
```

**How it differs from E and D:**
- **Logs bypass Cloud Logging.**
  - The predictor emits logs with the OTel logs SDK, and the deployment sets `disable_container_logging=True`.
  - There are no sinks, no Pub/Sub, and no `_Default` cost for endpoint logs.
  - **But** anything the SDK doesn't capture is invisible: uvicorn/startup failures before the SDK initialises, crash traces and OOM kills. Make sure the SDK's log handler is installed first thing at process start. If platform troubleshooting needs more, re-enable container logging temporarily for that endpoint.
- **No managed buffer.**
  - Pub/Sub's 7-day retention was the outage budget in D and E. Here, durability depends on the SDK's in-memory retry and the collector's sending queue.
  - Give the collector a **persistent queue** (`file_storage`) and 2+ replicas behind an internal load balancer. A full collector or Elasticsearch outage beyond the queue loses logs.
- **Predictor → collector path.**
  - Each predictor needs OTLP egress to the collector (an authenticated Cloud Run URL or an internal LB on a VPC the endpoint can reach). This is unverified for public Vertex endpoints.
  - Transport is **OTLP over HTTP on port 4318** (`http/protobuf`); gRPC (4317) isn't used. Model env vars:
    - `OTEL_EXPORTER_OTLP_PROTOCOL=http/protobuf`
    - `OTEL_EXPORTER_OTLP_ENDPOINT=https://<collector-host>:4318`
  - The SDK appends `/v1/logs` and `/v1/metrics`. The collector's `otlp` receiver enables only `protocols.http` (endpoint `0.0.0.0:4318`).
  - Every predictor image needs the OTel SDK (`opentelemetry-exporter-otlp-proto-http`).
- **Grafana stays on-prem.**
  - R5 is partly kept: the same Grafana and alerting, but metrics come from Elasticsearch rather than Prometheus/Thanos, so there's **no PromQL**.
  - Elasticsearch must expose an HTTPS endpoint to on-prem: Cloud Armor allow-list plus team-scoped API keys. This is the single firewall entry.
- **PII travels further.**
  - Log content leaves GCP on every Grafana query (rendered results only, not bulk egress).
  - Per-team indices and API keys keep R3.

| Pros | Cons |
|---|---|
| One pipeline and one store; no sink, Pub/Sub or Loki | **No managed buffer**: outage tolerance depends on the collector queue, not 7 days of Pub/Sub |
| Existing on-prem Grafana, one firewall entry | Code change in **every predictor** (OTel SDK for logs); platform-level failures before the SDK starts aren't captured |
| Little or no Cloud Logging cost if container logging is disabled | Metrics lose PromQL/Thanos; Elasticsearch becomes the metrics store too |
| Structured, consistent log schema from the SDK | Vertex → collector egress unverified; Elasticsearch publicly reachable (behind Cloud Armor) |

---

## 5. Evaluation

With the GCP-hosted OTel Collector, Elasticsearch and Kibana already in place, the log options compare as follows. Metrics are B in every row of the recommendation.

| Criterion | A Portal | B Grafana → CM | C Exporter → Thanos | D Sink → Loki | **E-hybrid on existing ELK** | E-direct |
|---|---|---|---|---|---|---|
| R1 Endpoint health | ✅ | ✅ | ✅ | n/a | ✅ via B (PromQL) | ⚠️ in Elasticsearch, no PromQL |
| R2 Log search | ⚠️ quota-bound | ❌ by design | ❌ | ✅ Grafana/Loki | ✅ Kibana (+ Grafana) | ✅ Kibana; ⚠️ misses crash/OOM/startup logs |
| R3 Per-team scoping | ✅ fine-grained | ⚠️ folder-level | ⚠️ folder-level | ✅ Loki tenant | ✅ data stream + role/Space per team | ✅ same |
| R4 One-way network | ✅ | ✅ | ✅ | ✅ | ✅ logs stay in GCP; users browse outbound | ⚠️ Vertex → collector egress unverified |
| R5 Reuse existing platforms | ❌ | ✅ | ✅ | ✅ if Loki exists | ✅ **existing collector, Elasticsearch, Kibana** | ✅ same |
| R6 Predictable cost | ✅ if cached | ⚠️ grows with viewers | ✅ fixed polling | ✅ volume-based + on-prem storage | ✅ Pub/Sub volume + incremental ES storage | ✅ lowest GCP cost |
| Outage buffer | n/a | n/a | n/a | ✅ Pub/Sub 7 d | ✅ Pub/Sub 7 d | ❌ collector queue only |
| Predictor code change | none | none | none | none | **none** | every predictor (OTel SDK) |
| PII location | GCP | n/a | n/a | **copied on-prem** | **stays in GCP** | stays in GCP |
| New components | portal UI | datasource | exporter | Alloy (+ Loki) | sink, topic, collector pipeline | collector endpoint, SDKs |
| Build effort | High | **Low** | Medium | Medium | **Low–Medium** | Medium–High |
| Freshness | ~2–3 min / s | ~2–3 min | 5–8 min | Seconds | Seconds | Seconds |

**Reading the table:**
- **B + E-hybrid** covers every requirement with the least new infrastructure. It reuses both existing platforms, keeps PII in GCP, and keeps Pub/Sub's buffer and full container-log capture.
- **D** loses its main advantage, reusing on-prem Loki, unless Loki already exists there. It also adds a bulk PII copy on-prem and an egress path. It remains the fallback.
- **E-direct** is the only option that gives up container-level logs and the outage buffer. That trade isn't justified while a Pub/Sub path is available.
- **C** remains the metrics upgrade path if B's read cost grows with viewers or retention beyond 24 months is required.

---

## 6. Cost analysis

All prices are list USD from the [Observability](https://cloud.google.com/stackdriver/pricing) and [Pub/Sub](https://cloud.google.com/pubsub/pricing) pricing pages, checked 2026-10-21 to 2026-10-23. Free allotments are **per billing account** unless noted, so other projects sharing the account consume them too.

### 6.1 Price points that matter

| Item | Price | Free per month |
|---|---|---|
| Built-in metric ingest | Free | n/a |
| Custom metric ingest, byte-based | $0.258/MiB → $0.151 → $0.061 | 150 MiB |
| Custom metric ingest, sample-based | $0.06 per M samples → … → $0.024 | None |
| **Monitoring reads** (`timeSeries.list`, PromQL) | **$0.50 per M time series returned**; each call is billed as max(series, 1) | 1M series |
| Monitoring writes | Free | n/a |
| Log storage in any log bucket | $0.50/GiB one-time, includes 30 days | 50 GiB **per project** |
| Log retention beyond 30 days | $0.01/GiB per month | n/a |
| Log Router routing, Log Analytics SQL, `entries.list` | Free (`entries.list` quota-bound) | n/a |
| Pub/Sub delivery (publish + subscribe) | $40/TiB, 1 KB minimum per request | 10 GiB |
| Pub/Sub Cloud Storage subscription | $50/TiB | None |
| Pub/Sub retained-message storage | $0.27/GiB per month | n/a |
| Alerting policies | $0.35 per metric reference per month, plus $0.50 per M points | From **2027-09-01** |

### 6.2 Metrics: the cost is in reads, not ingest

**Console reads are free.** Reads from client libraries and *"third-party tools like Grafana"* are billed. Moving users off the console turns a free read path into a metered one.

| Dashboard behaviour (20 endpoints × 3 models × 5 response codes × 6 metrics) | Series per refresh | Monthly read cost |
|---|---|---|
| Unaggregated, 30s refresh, 10 viewers | ~1,800 | **≈ $777** |
| `sum by (endpoint_id)`, 1-minute refresh, 10 viewers | 120 | **≈ $25** |
| Same, behind a single poller or cache (option C or a caching proxy) | 120 | **≈ $2** |

Server-side aggregation, minimum intervals and refresh limits are therefore **design requirements** for B, not optimisations. Track the cost with `monitoring.googleapis.com/billing/time_series_billed_for_queries_count`.

### 6.3 Logs

At 100 GiB/month of endpoint logs:

| Item | `_Default` only | + Pub/Sub → Elasticsearch (or Loki) | + full `_Default` exclusion |
|---|---|---|---|
| `_Default` storage | ≈ $25–50 (depends on the 50 GiB per-project free tier) | Same | $0 |
| Pub/Sub delivery (≈ 200 GiB of throughput) | n/a | ≈ $7.4 | ≈ $7.4 |
| GCS archive (Standard, regional) | n/a | ≈ $2 per month of data kept, before lifecycle tiering | Same |
| GCP → on-prem egress | n/a | None with Elasticsearch in GCP; unverified for Loki (D) | Same |
| Elasticsearch storage (existing cluster) | n/a | Incremental share, set by retention and ILM; unverified | Same |

Source reduction (§7.1) is worth more than any of these: the earlier work showed idle probe noise alone can dominate volume.

### 6.4 Custom metrics, if added

One series written once a minute is 43,200 points per month:

| Series | Byte-based | Sample-based |
|---|---|---|
| 1 scalar gauge | ≈ $0.085/mo | ≈ $0.003/mo |
| 1 histogram (10 non-zero buckets) | ≈ $0.85/mo | ≈ $0.03/mo |
| 1,000 histograms (e.g. 50 models × 20 label combinations) | **≈ $812/mo** | **≈ $31/mo** |

Cost is series × write frequency, so a bounded label allow-list is the main lever.

---

## 7. Log routing and exclusion strategy

Earlier drafts held two exclusion positions: exclude only the noise, which keeps the portal and Logs Explorer working, and exclude everything once Loki is trusted, which saves the most. They are reconciled here as **one staged strategy**.

### 7.1 How sinks and exclusions interact

```
Logging API ─► Log Router ─┬─► _Default sink  (exclusions) ─► _Default bucket   → Logs Explorer, entries.list
                           ├─► Pub/Sub sink   (own filter) ─► topic → Loki       → Grafana
                           ├─► GCS sink       (own filter) ─► archive bucket     → batch / BigQuery
                           └─► _Required sink (audit logs; cannot be excluded)
```

- **Sinks are independent.** An exclusion drops entries from *its own* sink's destination only. Every other sink still sees every entry. This repo measured it: with a `_Default` exclusion in place, the GCS sink kept receiving the excluded logs.
- **`entries.list` and Logs Explorer only read log buckets.** Anything excluded from `_Default` is invisible to option A, the Grafana Cloud Logging datasource, and the platform team in the console.
- **Exclusions cut storage, not ingestion.** Excluded entries still count toward the write quota and `billing/bytes_ingested`, so verify exclusions with Log Analytics `storage_bytes` instead.

### 7.2 The staged strategy

| Stage | Action | Effect |
|---|---|---|
| **1. Reduce at the source** | Set `LOG_LEVEL=WARNING` and `ACCESS_LOG=false` on the Model; log no payloads above DEBUG; optionally `disable_container_logging` for non-critical endpoints | Fewer bytes everywhere: cost, quotas, PII, Loki load, search quality |
| **2. Narrow exclusion on `_Default`** | Exclude only probe and access noise (by `logName`, found with `tools/logging-cost/discover.py`), optionally `sample(insertId, 0.9)` on high-volume INFO. Keep app logs, warnings and errors | Most of the storage saving; Logs Explorer and any API reader keep working |
| **3. Add the Pub/Sub sink (D)** | Use the same noise-free filter. This is a **separate sink** from the GCS sink | Loki becomes the user-facing log store |
| **4. Optional: full exclusion** | Once Loki has been the trusted system of record for a full retention cycle, exclude all Vertex endpoint logs from `_Default` | Removes the remaining $0.50/GiB. Logs Explorer is gone for these logs, and Loki plus GCS hold the only copies |

Rules that hold at every stage:
- Keep the Pub/Sub and GCS sink filters a superset of everything excluded, or those entries are lost everywhere.
- Scope every exclusion to `resource.type="aiplatform.googleapis.com/Endpoint"`, never project-wide.
- There are at most 50 exclusions per sink, so filter by resource type and labels, never per model.

### 7.3 Where GCS and BigQuery fit

| Need | Path | Notes |
|---|---|---|
| Archive, compliance | **GCS sink** (existing) | Hourly JSON batches, 2–3 h first delay. Routing is free; storage only. Cheaper than a Pub/Sub Cloud Storage subscription ($50/TiB) |
| Occasional SQL over archived logs | GCS → BigQuery external table or scheduled load job | Batch only. `YYYY/MM/DD` paths aren't Hive-partitioned; define `jsonPayload` as a `JSON` column |
| SQL analytics on fresh logs | Separate **sink → BigQuery dataset** (partitioned) | Near-real-time, no `entries.list` quota. BigQuery rates not verified here |
| Portal log view (if ever needed) | Stage 2 narrow exclusion with `entries.list`, or a dedicated log bucket with **per-model log views**, or BigQuery | The portal cannot read GCS interactively or reach on-prem Loki |

---

## 8. Data protection and residency

- **Every copy is a governed copy.** `_Default`, GCS, Pub/Sub and Elasticsearch (or Loki, under fallback D) each need IAM, retention and a location. A deletion request spans all four. With E-hybrid, every copy stays in GCP. Minimising at the source (§7.2, stage 1) is the only control that reduces all of them.
- **Least-privilege identities:**
  - Sink writer: `pubsub.publisher` on one topic.
  - On-prem consumer: `pubsub.subscriber` on one subscription.
  - Grafana: `monitoring.viewer`.
  - Nothing on-prem holds Logging roles.
- **Pub/Sub is a buffer, not a store:**
  - Messages are removed once every attached subscription acknowledges them.
  - Unacknowledged messages are kept 7 days by default (configurable from 10 minutes to 31 days). That retention is the **on-prem outage budget**; alert on `subscription/oldest_unacked_message_age`.
  - Replay (retain acknowledged messages or topic retention) costs $0.27/GiB per month. Use GCS for replay instead.
- **Pub/Sub location is not fixed by default:**
  - A topic isn't regional. Messages are stored in *"the nearest Google Cloud region"* to where they enter Google, and with no policy, any region is allowed. For a sink, Google is the publisher.
  - Set a **message storage policy** (`us-central1`) with `enforceInTransit`. On-prem Alloy must then use a regional endpoint.
- **Retention obligations:**
  - Pub/Sub (7 days) and `_Default` (30 days) aren't archives.
  - Any multi-year requirement is met by the GCS bucket's lifecycle and retention lock.
- **Per-team log access:** in Elasticsearch, use one data stream per team with index-privilege roles and a Kibana Space per team. Under fallback D, use one Loki tenant per team instead. Either way, R3 holds on the log path without per-request authorization code.

---

## 9. Custom model metrics (deferred)

Built-in metrics answer *"is the endpoint healthy?"*. Questions like *"is the model behaving?"* (class mix, confidence drift, feature null rates) need custom metrics emitted by the predictor. There's no named requirement yet, so this is deferred. When one appears:

| Path | Lands as | Verdict |
|---|---|---|
| **OTel SDK → OTLP gRPC → Telemetry API** (`telemetry.googleapis.com`) | `prometheus.googleapis.com/*`, sample-based | **Use this.** In-process, no infrastructure; visible in B and pollable by C |
| OTel SDK → OTel Collector (≥ 0.140.0) → Telemetry API | Same | Only if central filtering or relabeling is needed |
| Monitoring API `create_time_series` | `custom.googleapis.com/*`, byte-based | Fallback; ~25× costlier for histograms |
| Google `CloudMonitoringMetricsExporter` | `workload.googleapis.com/*`, byte-based | Google is steering users to OTLP |
| Ops Agent / Stackdriver Prometheus sidecar | `workload.` / `external.` | Not possible on Vertex (VM-only, or legacy) |

Guardrails:
- Labels come from a fixed allow-list (model, version, class), never request IDs or raw features.
- Export every 60s, off the request path.
- Histograms use few, explicit buckets.
- Vertex can't scrape a container `/metrics` endpoint and has no sidecar, so metrics must be pushed from inside the predictor. Env vars go on the Vertex **Model** (`Model.upload`).

A code sketch is in [Appendix C](#appendix-c--otlp-custom-metric-sketch).

---

## 10. Recommendation and rollout

**Recommended: B for metrics + E-hybrid on the existing GCP OTel Collector, Elasticsearch and Kibana for logs.**

| Phase | Scope | Exit criteria |
|---|---|---|
| **0. Source reduction** (days) | Stage 1 of §7.2 on every predictor: quiet logging, no payloads | `discover.py` shows probe/access lines gone; bytes/hour measured before and after |
| **1. Metrics via B** (days) | Read-only service account, firewall entry, on-prem Grafana datasource, a per-team dashboard template | On-prem Grafana shows p95 latency, prediction rate and error rate for one endpoint; billed series per refresh recorded |
| **2. Narrow exclusion** (1 day) | Stage 2 of §7.2 | `storage_bytes` drops; app errors are still visible in Logs Explorer |
| **3. Logs via the existing ELK** (≈ 1 week) | Topic with storage policy, sink, pull subscription for the collector; a `logs` pipeline on the existing collector (`googlecloudpubsub` + `googlecloudlogentry` encoding → `transform` team routing → `elasticsearch`); per-team data streams, roles and Kibana Spaces; backlog-age alert | Entries searchable in Kibana within seconds, each team sees only its own data stream, backlog drains, Elasticsearch storage/day matches the estimate |
| **4. Optional full exclusion** | Stage 4 of §7.2, after one full retention cycle on Elasticsearch | Explicit sign-off that Elasticsearch and GCS are the systems of record |
| **Later** | Custom metrics via OTLP; on-prem Grafana Elasticsearch datasource for log panels next to metrics; portal log view via the Elasticsearch API; move metrics to C if read cost or retention requires | A named need for each |

**Preconditions to confirm with the owners of the existing stack:**
- **Collector:** its build includes the contrib `googlecloudpubsub` receiver and the `googlecloudlogentry` encoding extension. The built-in `cloud_logging` encoding was removed in v0.132.0. Its service account can be granted `pubsub.subscriber` on the new subscription.
- **Elasticsearch:**
  - Capacity for the added ingest and retention.
  - Version ≥ 8.12 for the exporter's default `otel` mapping.
  - Per-team roles and Kibana Spaces are acceptable under the current licence.
- **Access:**
  - Kibana is already reachable from on-prem browsers with corporate SSO, or can be.
  - If on-prem Grafana panels are wanted, the Elasticsearch endpoint can also be reached from on-prem.

**Fallback:** if any precondition fails, use D (Alloy → on-prem Loki) for logs, with the Phase 3 GCP-side steps unchanged up to the subscription.

The portal links to the Kibana Space and Grafana dashboard for each model rather than rebuilding charts. A native in-portal log view, if wanted, queries Elasticsearch with per-team API keys, never `entries.list`.

---

## 11. Risks and open questions

| Risk / question | Impact | Mitigation / owner |
|---|---|---|
| GCP → on-prem **egress rate and proxy bandwidth** not yet known | Cost and throughput of D | Confirm with the network team before Phase 3 |
| **Loki limits** (ingestion rate, line size, streams per tenant) on the shared on-prem cluster | Rejected pushes leave messages in the backlog, then they're lost after 7 days | Agree tenant limits with the Loki owners; alert on backlog age |
| **Grafana auth to Google** from on-prem: service account key vs Workload Identity Federation | Key management burden | Prefer federation against the corporate IdP |
| Dashboard read cost grows with viewers | R6 | Aggregation, intervals, refresh limits; move to C if needed |
| Free allotments consumed by other projects on the billing account | Costs higher than estimated | Check billing account usage |
| Logs-based metrics and excluded entries | A log-derived alert might silently stop counting | Verify before relying on any user-defined logs-based metric |
| `sample()` and noise exclusions dropping something needed | Missing evidence in an incident | Never sample WARNING and above; GCS holds everything |
| **Shared Elasticsearch capacity and tenancy** | Endpoint logs compete with existing workloads; noisy ingest can affect other tenants | Agree ingest/retention budget and ILM policy with the owners; per-team data streams |
| **Existing collector** lacks the contrib receivers, or is shared with other pipelines | Phase 3 blocked, or a log pipeline failure affects other signals | Dedicated pipeline (or collector instance) for Vertex logs; pin the version; alert on backlog age |
| Kibana SSO / per-team Spaces under the current Elastic licence | R3 enforcement | Confirm with the Elastic owners before Phase 3 |

---

## 12. Verification status

**Verified** against primary sources (2026-10-21 to 2026-10-23) or measured:
- All prices in §6.1.
- Logging quotas (256 KiB entry, 60 req/min `entries.list`, 50 exclusions per sink, write rates).
- Pub/Sub quotas, retention range and location behaviour.
- The OTLP → `prometheus.googleapis.com` mapping and SKU.
- The Vertex built-in metric names.
- Sink independence under a `_Default` exclusion, measured in this repo.
- Grafana 13.2 Cloud Monitoring datasource with the PromQL query type, working locally.

**Not yet verified:**

| Item | How to verify |
|---|---|
| GCP → on-prem egress pricing | Billing and network team |
| BigQuery storage and query rates | Pricing page |
| Exact `stackdriver_exporter` project flag | `--help` on the deployed release |
| `gcloud` flag name for `enforceInTransit` | `--help` |
| OTLP gRPC endpoint form `telemetry.googleapis.com:443` | First export test |
| Structured JSON stdout → `jsonPayload` + `severity` on Vertex | `discover.py` |
| Alloy's `__gcp_*` relabel label names | Alloy UI |
| Built-in metric ingest delay (~2–3 min) | Spike |
| Option E: Elasticsearch hosting cost and sizing; document-level security licensing | Elastic pricing, owning team |
| Option E: `googlecloudlogentry` encoding output mapping in Elasticsearch; Monitoring receiver behaviour (alpha) | Collector spike against a test index |
| Option E: Vertex predictor egress to a GCP-hosted collector (custom OTLP metrics) | Deploy test |
| Option E: Kibana SSO (SAML/OIDC) licensing; Logstash/Elastic Agent GCP inputs | Elastic docs, owning team |
| Option E: corporate egress IP ranges for Cloud Armor; ADA SSO as an OIDC/SAML IdP for Grafana | Network and identity teams |

---

## Appendix A — Option D setup

Steps 1–5 run in GCP; steps 6–8 run on-prem.

1. **Topic:**
   - Create `vertex-endpoint-logs`.
   - Set a message storage policy restricted to `us-central1`, with in-transit enforcement.
2. **Sink:**
   - Destination is the topic.
   - Filter `resource.type="aiplatform.googleapis.com/Endpoint"`, minus probe and access `logName`s.
   - No `timestamp` or `location` predicates, so the rule keeps matching.
3. **Sink identity:** grant its writer identity `roles/pubsub.publisher` on this topic only.
4. **Subscription:**
   - Create `vertex-endpoint-logs-onprem` as a **pull** subscription (push would need GCP to reach on-prem).
   - Ack deadline 60s, retention 7 days.
   - Optionally add a dead-letter topic.
5. **Consumer identity:** create a dedicated service account with `roles/pubsub.subscriber` on this subscription only. Prefer federation; otherwise a rotated key in the on-prem secret store.
6. **Firewall:** allow outbound to `pubsub.googleapis.com` and `oauth2.googleapis.com`, or the `private.googleapis.com` /30.
7. **Alloy:** `loki.source.gcplog` in pull mode → relabel → `loki.write` with a team tenant.
8. **Grafana and alerting:**
   - Add the Loki datasource and per-team folders.
   - Alert on `subscription/oldest_unacked_message_age`, dead-letter growth, and sink `exports/error_count`.

```bash
gcloud pubsub topics create vertex-endpoint-logs \
  --message-storage-policy-allowed-regions=us-central1   # + in-transit enforcement flag (confirm name with --help)

gcloud logging sinks create vertex-endpoint-logs-to-pubsub \
  pubsub.googleapis.com/projects/$PROJECT_ID/topics/vertex-endpoint-logs \
  --log-filter='resource.type="aiplatform.googleapis.com/Endpoint"'   # tighten with noisy logNames excluded

gcloud pubsub topics add-iam-policy-binding vertex-endpoint-logs \
  --member="$(gcloud logging sinks describe vertex-endpoint-logs-to-pubsub --format='value(writerIdentity)')" \
  --role=roles/pubsub.publisher

gcloud pubsub subscriptions create vertex-endpoint-logs-onprem \
  --topic=vertex-endpoint-logs --ack-deadline=60 --message-retention-duration=7d

gcloud pubsub subscriptions add-iam-policy-binding vertex-endpoint-logs-onprem \
  --member=serviceAccount:onprem-log-reader@$PROJECT_ID.iam.gserviceaccount.com \
  --role=roles/pubsub.subscriber
```

```alloy
loki.source.gcplog "vertex" {
  pull {
    project_id             = "mle-project-496007"
    subscription           = "vertex-endpoint-logs-onprem"
    use_incoming_timestamp = true
  }
  forward_to    = [loki.write.onprem.receiver]
  relabel_rules = loki.relabel.vertex.rules
}

loki.relabel "vertex" {
  forward_to = []
  rule {
    source_labels = ["__gcp_resource_labels_endpoint_id"]
    target_label  = "endpoint_id"
  }
  rule {
    source_labels = ["__gcp_severity"]
    target_label  = "level"
  }
  rule {
    source_labels = ["__gcp_logname"]
    target_label  = "log_name"
  }
}

loki.write "onprem" {
  endpoint { url = "https://loki.internal.example/loki/api/v1/push" }
}
```

## Appendix B — Option B starter queries

Use the Google Cloud Monitoring datasource with query type **PromQL**, **one expression per query row**. Two expressions in one box fail with `unexpected <aggr:sum>`.

```promql
# p95 latency per endpoint
histogram_quantile(0.95, sum by (le, endpoint_id) (rate(aiplatform_googleapis_com:prediction_online_prediction_latencies_bucket{monitored_resource="aiplatform.googleapis.com/Endpoint"}[5m])))
```
```promql
# prediction rate per endpoint
sum by (endpoint_id) (rate(aiplatform_googleapis_com:prediction_online_prediction_count{monitored_resource="aiplatform.googleapis.com/Endpoint"}[5m]))
```

Metric names map from Cloud Monitoring as follows: `.` becomes `_`, the first `/` becomes `:`, and the remaining `/` become `_`. Start with a bare selector to confirm a name resolves.

## Appendix C — OTLP custom metric sketch

Adapted from Google's Python sample; not yet run here.

```python
# deps: opentelemetry-sdk, opentelemetry-exporter-otlp-proto-grpc,
#       opentelemetry-resourcedetector-gcp, google-auth, grpcio
import google.auth, google.auth.transport.grpc, google.auth.transport.requests, grpc
from opentelemetry import metrics
from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import get_aggregated_resources
from opentelemetry.resourcedetector.gcp_resource_detector import GoogleCloudResourceDetector

credentials, _ = google.auth.default()
channel_creds = grpc.composite_channel_credentials(
    grpc.ssl_channel_credentials(),
    grpc.metadata_call_credentials(
        google.auth.transport.grpc.AuthMetadataPlugin(credentials, google.auth.transport.requests.Request())),
)
reader = PeriodicExportingMetricReader(OTLPMetricExporter(credentials=channel_creds), export_interval_millis=60_000)
resource = get_aggregated_resources([GoogleCloudResourceDetector(raise_on_error=False)])
metrics.set_meter_provider(MeterProvider(metric_readers=[reader], resource=resource))

meter = metrics.get_meter("xgboost-census")
predicted_class = meter.create_counter("model_predictions", description="Predictions by class")
confidence = meter.create_histogram("model_confidence", description="Top-class probability")
# in /predict:  predicted_class.add(1, {"model": MODEL_NAME, "class": label})
#               confidence.record(prob, {"model": MODEL_NAME})
```

**Model env vars:**
- `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT=https://telemetry.googleapis.com:443`
- `OTEL_RESOURCE_ATTRIBUTES=gcp.project_id=<PROJECT>,service.name=xgboost-census`. `gcp.project_id` is required.

**Serving service account roles:** `roles/telemetry.writer` and `roles/serviceusage.serviceUsageConsumer`.

**Limits:** ≤200 points per export request.

## Appendix D — Local validation harness

A laptop stands in for on-prem: outbound to Google APIs only. It runs Docker Compose with Grafana, Loki and Alloy, and a **dedicated read-only test service account** whose key is kept outside the repo and deleted afterwards.

- **B:** add the Google Cloud Monitoring datasource with a JWT key, then run the Appendix B queries in Explore.
- **D:** point Alloy at the test subscription, send a few predictions, and query `{endpoint_id="<ID>"}` in Loki.
- **Cost check:** read `billing/time_series_billed_for_queries_count` after an hour of dashboard use.
- **Teardown:** `docker compose down -v`, delete the test key, and remove the test sink, topic and subscription.

## References

- Cloud Logging API: https://docs.cloud.google.com/logging/docs/reference/api-overview · Quotas: https://docs.cloud.google.com/logging/quotas · GCS export format: https://docs.cloud.google.com/logging/docs/export/storage
- Cloud Monitoring API v3: https://docs.cloud.google.com/monitoring/api/v3 · Built-in metrics: https://docs.cloud.google.com/monitoring/api/metrics_gcp_a_b · User-defined metrics: https://docs.cloud.google.com/monitoring/custom-metrics · Quotas: https://docs.cloud.google.com/monitoring/quotas
- Observability pricing: https://cloud.google.com/stackdriver/pricing · Pub/Sub pricing: https://cloud.google.com/pubsub/pricing
- OTLP metrics: https://docs.cloud.google.com/stackdriver/docs/otlp/overview · SDK export: https://docs.cloud.google.com/stackdriver/docs/otlp-metrics/use-sdks · Collector: https://docs.cloud.google.com/stackdriver/docs/otlp-metrics/deploy-collector · Telemetry API: https://docs.cloud.google.com/stackdriver/docs/reference/telemetry/overview
- Pub/Sub subscription properties: https://docs.cloud.google.com/pubsub/docs/subscription-properties · Location restriction: https://docs.cloud.google.com/pubsub/docs/resource-location-restriction · Quotas: https://docs.cloud.google.com/pubsub/quotas
- Grafana Cloud Monitoring datasource: https://grafana.com/docs/grafana/latest/datasources/google-cloud-monitoring/ · Alloy `loki.source.gcplog`: https://grafana.com/docs/alloy/latest/reference/components/loki/loki.source.gcplog/
- OTel Collector contrib (option E): Pub/Sub receiver https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/receiver/googlecloudpubsubreceiver/README.md · Cloud Monitoring receiver https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/receiver/googlecloudmonitoringreceiver/README.md · Elasticsearch exporter https://github.com/open-telemetry/opentelemetry-collector-contrib/blob/main/exporter/elasticsearchexporter/README.md
