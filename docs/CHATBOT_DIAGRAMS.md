# Diagrammes du systeme chatbot

Ce document donne une vue visuelle complete du chatbot MASI Risk Engine: interface, API, orchestration, RAG vectoriel, contexte numerique, LLM local et garde-fous.

## 1. Vue globale du chatbot

```mermaid
flowchart LR
    User(["Utilisateur"]) --> UI["Dashboard web<br/>Chat panel"]
    UI --> API["FastAPI<br/>/chat/ask<br/>/chat/ask/stream"]
    API --> Service["Chat service<br/>orchestrateur"]

    Service --> Validate["Validation question<br/>historique recent"]
    Validate --> Intent["Embedding intent router<br/>centroid similarity<br/>help, forecast, backtest,<br/>definition, model, data"]
    Intent --> Policy["Response policy<br/>autorise ou bloque le LLM"]
    Intent --> Context["Routed context builder"]

    Context --> DashState["Etat exact du dashboard<br/>current_dashboard_state"]
    Context --> Numeric["Contexte numerique<br/>forecast, backtest, metadata"]
    Context --> RAG["RAG vectoriel Chroma<br/>documents techniques"]

    DashState --> Prompt["Prompt builder"]
    Numeric --> Prompt
    RAG --> Prompt
    Policy --> Prompt
    Validate --> Prompt

    Prompt --> Gate{"LLM autorise ?"}
    Gate -- "Non" --> Direct["Reponse directe controlee<br/>refus conseil financier<br/>guidance deterministe"]
    Gate -- "Oui" --> LLM["Ollama local<br/>qwen2.5:3b"]

    LLM --> Repair["Answer repair<br/>corrections MASI"]
    Repair --> Guardrails["Guardrails<br/>anti hallucination<br/>anti conseil<br/>VaR/ES"]
    Direct --> Payload["Payload API"]
    Guardrails --> Payload
    Payload --> UI

    classDef user fill:#f8f9fa,stroke:#222,stroke-width:1px,color:#111;
    classDef app fill:#d8f0d2,stroke:#3c6e3c,stroke-width:1px,color:#111;
    classDef control fill:#d9e8ff,stroke:#315f9e,stroke-width:1px,color:#111;
    classDef rag fill:#fff2cc,stroke:#997a00,stroke-width:1px,color:#111;
    classDef safety fill:#ffd9d9,stroke:#9e3131,stroke-width:1px,color:#111;

    class User user;
    class UI,API,Service,Prompt,Payload app;
    class Validate,Intent,Policy,Context,Gate control;
    class DashState,Numeric,RAG rag;
    class Direct,Repair,Guardrails safety;
```

## 2. Construction de la base RAG vectorielle

```mermaid
flowchart LR
    Docs["Documents Markdown<br/>backend/chatbot/rag/docs"] --> Clean["Nettoyage Markdown"]
    Clean --> SplitHeaders["Split par titres<br/>#, ##, ###"]
    SplitHeaders --> ChildChunks["Child chunks<br/>environ 200 tokens<br/>overlap 40"]
    ChildChunks --> Refine["Refine chunks longs<br/>Recursive splitter"]
    Refine --> Embed["Embedding model<br/>sentence-transformers/all-MiniLM-L6-v2"]
    Embed --> Vectors["Vecteurs normalises"]
    Vectors --> Chroma[("Chroma DB locale<br/>backend/chatbot/rag/vector_db")]

    BuildCmd["Commande<br/>python -m backend.chatbot.rag.build_index"] --> Docs

    classDef source fill:#f8f9fa,stroke:#222,color:#111;
    classDef process fill:#d8f0d2,stroke:#3c6e3c,color:#111;
    classDef vector fill:#d9e8ff,stroke:#315f9e,color:#111;
    classDef db fill:#fff2cc,stroke:#997a00,color:#111;

    class BuildCmd,Docs source;
    class Clean,SplitHeaders,ChildChunks,Refine process;
    class Embed,Vectors vector;
    class Chroma db;
```

## 3. Retrieval RAG pendant une question

```mermaid
flowchart LR
    Question["Question utilisateur"] --> Intent["Intent detectee"]
    Intent --> Route{"Route RAG requise ?"}

    Route -- "definition/model/data" --> EmbedQ["Embedding de la question"]
    Route -- "forecast/backtest pur" --> NoRAG["Pas de RAG<br/>contexte numerique suffit"]

    EmbedQ --> Chroma[("Chroma vector_db")]
    Chroma --> Search["Similarity search<br/>top-k passages"]
    Search --> Format["Format passages<br/>source + section + contenu"]
    Format --> RAGContext["Contexte documentaire"]

    NoRAG --> Prompt["Prompt final"]
    RAGContext --> Prompt
    Numeric["Contexte numerique<br/>forecast/backtest/metadata"] --> Prompt
    Dash["Etat dashboard prioritaire"] --> Prompt
    Prompt --> LLM["LLM local"]
    LLM --> Answer["Reponse controlee"]

    classDef control fill:#d9e8ff,stroke:#315f9e,color:#111;
    classDef rag fill:#fff2cc,stroke:#997a00,color:#111;
    classDef app fill:#d8f0d2,stroke:#3c6e3c,color:#111;
    classDef safety fill:#ffd9d9,stroke:#9e3131,color:#111;

    class Question,Intent,Route control;
    class EmbedQ,Chroma,Search,Format,RAGContext rag;
    class Numeric,Dash,Prompt,LLM app;
    class NoRAG,Answer safety;
```

## 4. Routage du contexte par intention

```mermaid
flowchart TD
    Q["Question utilisateur"] --> Embed["Light embedding<br/>all-MiniLM-L6-v2"]
    Embed --> Centroids["Centroïdes d'intentions<br/>exemples annotés"]
    Centroids --> Intent["Intent router<br/>cosine similarity"]
    Intent --> Help["help_request"]
    Intent --> Forecast["forecast_query"]
    Intent --> Backtest["backtest_query"]
    Intent --> Strategy["strategy_query"]
    Intent --> Definition["definition_query"]
    Intent --> Model["model_query"]
    Intent --> Data["data_query"]
    Intent --> Out["out_of_scope"]

    Help --> Guided["Guidance deterministe<br/>si l'utilisateur accepte l'aide"]
    Help --> ForecastCtx["Forecast context"]
    Help --> BacktestCtx["Backtest context"]

    Forecast --> ForecastCtx
    Backtest --> BacktestCtx
    Strategy --> BacktestCtx

    Definition --> StaticRAG["Static RAG Chroma"]
    Model --> StaticRAG
    Data --> StaticRAG
    Model --> Metadata["Model metadata"]
    Data --> Metadata

    Out --> Refusal["Reponse de cadrage<br/>retour au MASI dashboard"]

    ForecastCtx --> Prompt["Prompt ou reponse directe"]
    BacktestCtx --> Prompt
    StaticRAG --> Prompt
    Metadata --> Prompt
    Guided --> Payload["Payload API"]
    Prompt --> Payload
    Refusal --> Payload

    classDef intent fill:#d9e8ff,stroke:#315f9e,color:#111;
    classDef ctx fill:#fff2cc,stroke:#997a00,color:#111;
    classDef direct fill:#ffd9d9,stroke:#9e3131,color:#111;
    classDef output fill:#d8f0d2,stroke:#3c6e3c,color:#111;

    class Intent,Help,Forecast,Backtest,Strategy,Definition,Model,Data,Out intent;
    class ForecastCtx,BacktestCtx,StaticRAG,Metadata ctx;
    class Guided,Refusal direct;
    class Prompt,Payload output;
```

## 5. Garde-fous et validation finale

```mermaid
flowchart LR
    Raw["Reponse brute LLM"] --> Repair["Repair known MASI errors"]
    Repair --> Validate["Validate response"]

    Validate --> Nums{"Chiffres absents<br/>du contexte ?"}
    Validate --> Advice{"Conseil achat/vente ?"}
    Validate --> VarES{"Confusion VaR / ES ?"}
    Validate --> Certainty{"Certitude excessive ?"}
    Validate --> Dates{"Dates relatives<br/>inventees ?"}
    Validate --> Repetition{"Repetition ou dump ?"}

    Nums -- "Oui" --> Fallback["Correction ou fallback"]
    Advice -- "Oui" --> Refuse["Refus conseil financier"]
    VarES -- "Oui" --> Explain["Correction definition VaR/ES"]
    Certainty -- "Oui" --> Fallback
    Dates -- "Oui" --> Fallback
    Repetition -- "Oui" --> Fallback

    Nums -- "Non" --> OK["Reponse valide"]
    Advice -- "Non" --> OK
    VarES -- "Non" --> OK
    Certainty -- "Non" --> OK
    Dates -- "Non" --> OK
    Repetition -- "Non" --> OK

    Fallback --> Final["Reponse finale API"]
    Refuse --> Final
    Explain --> Final
    OK --> Final

    classDef llm fill:#d8f0d2,stroke:#3c6e3c,color:#111;
    classDef check fill:#d9e8ff,stroke:#315f9e,color:#111;
    classDef risk fill:#ffd9d9,stroke:#9e3131,color:#111;
    classDef final fill:#fff2cc,stroke:#997a00,color:#111;

    class Raw,Repair llm;
    class Validate,Nums,Advice,VarES,Certainty,Dates,Repetition check;
    class Fallback,Refuse,Explain risk;
    class OK,Final final;
```

## 6. Sequence API streaming

```mermaid
sequenceDiagram
    participant U as Utilisateur
    participant D as Dashboard
    participant A as FastAPI /chat/ask/stream
    participant S as Chat service
    participant R as Chroma retriever
    participant L as Ollama
    participant G as Guardrails

    U->>D: Question
    D->>A: POST question + historique + etat dashboard
    A->>S: stream_masi_chatbot()
    S->>S: validation + intent + policy
    alt question guidee ou conseil refuse
        S-->>A: reponse directe controlee
    else LLM autorise
        S->>R: retrieve relevant context
        R-->>S: passages pertinents
        S->>L: prompt controle
        L-->>S: chunks de reponse
        S->>G: repair + validation
        G-->>S: reponse finale
    end
    S-->>A: delta final
    S-->>A: done + metadata
    A-->>D: NDJSON stream
    D-->>U: Reponse affichee
```
