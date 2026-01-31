# FraudShield AI  
**A Real-Time, Explainable, Secure Fraud Decision System for UPI & Digital Payments in India**

---

## 1. Project Title

**FraudShield AI**

---

## 2. Abstract

**FraudShield AI** is a production-oriented, real-time fraud decision platform designed specifically for India’s UPI and digital payment ecosystem. The system addresses critical challenges such as extreme class imbalance, evolving fraud patterns, sub-second decision requirements, and strict regulatory demands for explainability and auditability.

The solution combines **classical machine learning models**, **unsupervised anomaly detection**, and a **multi-agent decision orchestration layer** to deliver accurate and low-latency fraud decisions. To meet regulatory and human interpretability requirements, the system integrates **Large Language Models (LLMs)** through a **Retrieval-Augmented Generation (RAG)** pipeline supported by **embeddings and a vector database** containing RBI circulars, NPCI UPI rules, and curated fraud case knowledge.

An **Agentic RAG architecture** is employed, where bounded AI agents perform risk scoring, policy enforcement, explanation generation, analyst assistance, and model drift monitoring. **Model Context Protocol (MCP)** principles are applied to strictly control and sanitize the context shared with LLMs, ensuring no raw transaction data or PII is exposed. As part of the hands-on **capstone mini build**, FraudShield AI demonstrates a complete, secure, and explainable end-to-end fraud decision system suitable for real-world financial environments.

---

## 3. Problem Statement

India processes billions of UPI transactions annually, with fraud accounting for less than 1% of total volume. This creates a uniquely difficult problem characterized by:

- Extreme class imbalance  
- Millisecond-level decision constraints  
- High cost of false positives impacting customer trust  
- Constantly evolving fraud techniques (QR scams, SIM swaps, mule accounts)  
- Regulatory requirements for explainable and auditable decisions (RBI, NPCI)

Most existing systems are batch-oriented, credit-card focused, or rely on black-box models that fail to meet real-time, compliance, and deployment requirements of India’s UPI ecosystem.

---

## 4. Objectives

- Detect fraudulent UPI transactions in real time  
- Minimize false positives while maintaining high fraud recall  
- Identify previously unseen fraud patterns using anomaly detection  
- Provide regulator- and human-friendly explanations  
- Enforce RBI and NPCI policy compliance programmatically  
- Support analyst investigation and auditing  
- Demonstrate safe and controlled use of LLMs in financial systems  

---

## 5. Technologies & Concepts Used

- **Large Language Models (LLMs)** – Offline, open-source models used only for explanation  
- **Embeddings & Vector Databases** – Semantic retrieval of regulatory and fraud knowledge  
- **Retrieval-Augmented Generation (RAG)** – Grounded explanation generation  
- **Agentic RAG** – Multiple bounded AI agents with single responsibilities  
- **Model Context Protocol (MCP)** – Strict control over LLM context and inputs  
- **Classical Machine Learning** – Random Forest, XGBoost, Neural Networks  
- **Unsupervised Learning** – AutoEncoder-based anomaly detection  
- **Security & Compliance** – RBAC, encryption, audit logging  

---

## 6. System Architecture / Workflow

### High-Level Workflow

1. A UPI transaction enters the system  
2. Feature engineering and input validation are performed  
3. Supervised ML ensemble computes fraud probability  
4. AutoEncoder evaluates behavioral anomaly score  
5. Decision Orchestration Agent determines the final action  
6. RAG-based Explanation Agent generates grounded explanations (if required)  
7. Text-to-Speech converts approved explanations for human consumption  
8. Decisions, explanations, and logs are securely stored for auditing  

LLMs are **never** part of the real-time decision path and operate only under controlled context following MCP principles.

---

## 7. Implementation Details

- **Backend**: Python-based microservices  
- **ML Stack**: Scikit-learn, XGBoost, PyTorch  
- **Vector Store**: FAISS / Chroma (offline, local)  
- **LLM**: Mistral 7B / Llama 3.x (offline inference)  
- **Agent Layer**: Rule-bounded orchestration agents  
- **Security**: OAuth 2.0, JWT, Role-Based Access Control  
- **Explainability**: SHAP values + RAG-generated explanations  

### Core Modules
- Feature Engineering Layer  
- Supervised Ensemble ML Engine  
- Anomaly Detection Module  
- Multi-Agent Decision Orchestration Layer  
- RAG Explanation Pipeline  
- Audit, Logging, and Monitoring System  

---

## 8. Features

- Real-time fraud decisioning with sub-second latency  
- Ensemble-based fraud risk scoring  
- Detection of unknown fraud via anomaly detection  
- Agent-driven decision orchestration  
- RAG-based, regulation-grounded explanations  
- Text-to-Speech for analyst accessibility  
- Policy-compliant soft and hard blocking  
- Secure, tamper-proof audit logs  

---

## 9. Usage Instructions

1. Clone the repository  
2. Install dependencies using `requirements.txt`  
3. Initialize the vector database with curated regulatory and fraud documents  
4. Start the backend service  
5. Submit transactions via API or test scripts  
6. View decisions, explanations, and logs through the interface  

---

## 10. Capstone / Mini Build Description

As part of the hands-on capstone mini build, FraudShield AI demonstrates:

- A functional real-time fraud scoring pipeline  
- Embedding-based retrieval of RBI and NPCI documents  
- RAG-generated explanations grounded in retrieved context  
- Agentic orchestration across ML, policy, and explanation layers  
- Controlled LLM usage aligned with Model Context Protocol principles  

This mini build validates the practical application of LLMs, embeddings, RAG, agents, and context control in a high-stakes financial domain.

---

## 11. Future Scope

- Integration with UPI sandbox or banking simulators  
- Multilingual explanation support for Indian languages  
- Online learning and drift-aware retraining  
- Graph-based fraud detection for mule networks  
- Voice-based customer-facing explanations  

---

## 12. Contributors

- **Ananya Bachchhav** – Team Lead, Backend Development  
- **Ronan Fonseca** – Backend Development, Documentation  
- **Prathamesh Patil** – Documentation  
- **Kshitija Hire** – Frontend Development  

---

## 13. References (Optional)

- RBI Circulars on Digital Payment Security  
- NPCI UPI Procedural Guidelines  
- Fraud Detection Research Literature  
- SHAP Documentation  
- FAISS / Chroma Documentation  