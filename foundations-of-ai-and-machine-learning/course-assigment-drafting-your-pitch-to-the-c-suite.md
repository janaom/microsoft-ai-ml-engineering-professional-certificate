# PITCH: Adopting Google Cloud Vertex AI for ML Model Deployment

# INTRODUCTION

Since Vinted's infrastructure is already on Google Cloud Platform, we have a unique opportunity to leverage a fully integrated ML deployment solution. As we continue to scale globally, our recommendation engine, fraud detection, and pricing optimization models are scattered across different systems, slowing innovation and increasing operational costs. Selecting the right deployment platform is essential to maintaining our competitive edge in the dynamic resale marketplace.

# PROPOSED SOLUTION & COST

I recommend adopting Google Cloud Vertex AI, a managed machine learning platform that handles model training, deployment, and monitoring in one place.

Annual Cost Estimate: €130K-170K (including infrastructure, licensing, and training)

Current scattered approach: €220K+ (inefficient tooling, staff overhead, redundant infrastructure)

Net savings: €50K-90K annually, plus improved operational efficiency

# BUSINESS BENEFITS

- Faster Time-to-Market (30% improvement)

ML engineers currently spend 40% of time on infrastructure management. Vertex AI automates deployment, freeing engineers to focus on model improvements. We can deploy new recommendation features weekly instead of monthly.


- Cost Efficiency & Scalability (20-30% cost reduction)

Vertex AI auto-scales with demand—we pay only for what we use. During peak holiday shopping, we scale automatically. During slow periods, costs drop. This is far better than our current fixed infrastructure spending.


- Better Business Outcomes (5-15% improvement)

Our fraud detection and pricing models can be A/B tested faster, leading to better conversion rates and reduced chargebacks. Early analysis suggests a 5-15% improvement in recommendation accuracy within 6 months.


- Security & Compliance (Meets GDPR/Data Protection)

Vertex AI includes built-in encryption, audit logging, and data governance tools. With our user base in EU/UK, compliance is non-negotiable. This platform handles it automatically, reducing legal/security team overhead.


- Seamless GCP Integration

Since our entire infrastructure already runs on GCP (BigQuery, Dataflow, Cloud Storage), Vertex AI integrates natively with zero additional tooling. No new vendors, no fragmented ecosystems—one unified platform.



# RISK MITIGATION

Risk 1: Vendor Lock-in

Mitigation: We maintain model code in open formats (ONNX, Scikit-learn). Migration would take 2-3 months if needed, but Vertex AI's cost savings and GCP integration make switching unlikely.

Risk 2: Implementation Timeline

Mitigation: Phased rollout—start with 2-3 lower-risk models (pricing engine), then migrate recommendation engine. Timeline: 3 months, zero disruption to production.

Risk 3: Team Training

Mitigation: Google provides 2 weeks of onboarding training included. Our team is already familiar with GCP services. Ramp-up time: 1-2 weeks for full productivity.

Risk 4: Cost Overruns

Mitigation: Set spend alerts and caps. Monitor costs weekly. We've budgeted conservatively—actual spend likely 10-20% lower than estimate.

# CALL TO ACTION

I recommend we approve a 3-month pilot program with Vertex AI:

Deploy 3 models to test the platform

Measure deployment time, cost, and model performance

Decision point: Full rollout or alternative approach

### Timeline:

Week 1: Approve & contract with Google
Week 2-4: Setup & team training
Week 5-12: Pilot deployment
Week 13: Board review & decision

Expected Outcome: Proven ROI, faster model deployment, and a solid foundation for scaling our ML capabilities globally.

I recommend we move forward with the pilot. This positions Vinted to compete more effectively and deliver better personalized experiences to our users.
