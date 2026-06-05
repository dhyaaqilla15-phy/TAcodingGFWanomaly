# Flowchart Project AIS/GFW Gear + Spoofing + Go-Dark

Versi ini sudah menghapus command `predict`. Tahap akhir untuk laporan skripsi adalah `eval`, karena confusion matrix, metrics, dan tabel prediksi evaluasi sudah dibuat di sana.

```mermaid
flowchart LR
  classDef startEnd fill:#1f6feb,stroke:#c9d1d9,color:#ffffff,stroke-width:2px;
  classDef process fill:#111827,stroke:#9ca3af,color:#f9fafb,stroke-width:1.5px;
  classDef decision fill:#3f3f00,stroke:#d1d5db,color:#f9fafb,stroke-width:1.5px;
  classDef io fill:#0b3d1e,stroke:#d1d5db,color:#f9fafb,stroke-width:1.5px;

  A([START: python main.py cmd]):::startEnd
  B{Pilih command}:::decision
  A --> B

  subgraph G[Gear Classification]
    G1[preprocess --task gear]:::process
    G2[train model gear]:::process
    G3[eval model gear]:::process
    G4[/confusion_matrix.png<br/>confusion_matrix_normalized.png<br/>per_vessel_predictions.csv<br/>eval_summary.json/]:::io
    G1 --> G2 --> G3 --> G4
  end

  subgraph S[Spoofing Pipeline]
    S1[make_spoofing]:::process
    S2[plot_spoofing / heatmap_spoofing]:::process
    S3[preprocess --task spoofing]:::process
    S4[train model spoofing]:::process
    S5[eval model spoofing]:::process
    S6[/spoofed_all.csv<br/>confusion_matrix.png<br/>eval_summary.json/]:::io
    S1 --> S2 --> S3 --> S4 --> S5 --> S6
  end

  subgraph D[Go-Dark Pipeline]
    D1[make_godark]:::process
    D2[plot_godark / heatmap_godark]:::process
    D3[preprocess --task godark]:::process
    D4[train model go-dark]:::process
    D5[eval model go-dark]:::process
    D6[/godark_all.csv<br/>events/*.csv<br/>hidden_truth/*.csv<br/>confusion_matrix.png<br/>eval_summary.json/]:::io
    D1 --> D2 --> D3 --> D4 --> D5 --> D6
  end

  subgraph U[Utility]
    U1[plot / plot_all]:::process
    U2[heatmap]:::process
    U3[/PNG trajectory / heatmap/]:::io
    U1 --> U3
    U2 --> U3
  end

  B -->|gear| G1
  B -->|spoofing| S1
  B -->|go-dark| D1
  B -->|visualisasi| U1
  B -->|visualisasi| U2

  Z([END: hasil eval siap untuk skripsi]):::startEnd
  G4 --> Z
  S6 --> Z
  D6 --> Z
  U3 --> Z
```
