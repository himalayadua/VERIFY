# NovaAI Internal Wiki — Company Overview & Reference

> **Last updated:** March 2026 | Maintained by People Operations  
> This document is for internal use only. Do not share externally.

---

## 1. Company Overview

NovaAI was founded in **2021** by **Dr. Mara Chen** and **Lucas Ferreira** in **San Francisco, California**. The company is headquartered at **450 Market Street, Suite 1200, San Francisco, CA 94105**. NovaAI's mission is to make enterprise-grade AI infrastructure accessible to mid-market companies that lack the resources to build in-house ML teams.

As of Q1 2026, NovaAI employs **312 full-time employees** across its San Francisco HQ, a **London office** (opened in 2023), and a **Singapore office** (opened in 2024). The company is privately held and reached a **$1.4B valuation** following its Series C round in November 2025, led by Horizon Ventures. Total funding to date stands at **$210M**.

**Current CEO:** Dr. Mara Chen  
**CTO:** Lucas Ferreira  
**CFO:** Priya Nambiar (joined 2022)  
**COO:** James Whitfield (joined 2023)

The fiscal year runs **January 1 – December 31**. The company's primary verticals are **financial services, healthcare, and logistics**.

---

## 2. Department Structure

### 2.1 Engineering
- **VP of Engineering:** Sofia Andersson
- Divided into four sub-teams: Platform Infrastructure, ML Systems, Product Engineering, and Security Engineering.
- Total headcount: 118 engineers as of Q1 2026.
- Platform Infrastructure lead: **Raj Mehta** — responsible for cloud infra (AWS primary, GCP secondary).
- ML Systems lead: **Dr. Yuki Tanaka** — oversees model training pipelines and evaluation frameworks.
- Product Engineering lead: **Carlos Rivera** — embedded squads aligned to each product line.
- Security Engineering lead: **Ananya Bose** — handles SOC 2 compliance and pen testing cycles.

### 2.2 Product
- **VP of Product:** Oliver Haines
- Three product managers (PMs), each owning one product line. A fourth PM role is currently open (req #PM-2026-04).
- UX/Design is housed under Product and led by **Lena Park** (Head of Design, 6 designers total).

### 2.3 Sales & Revenue
- **Chief Revenue Officer (CRO):** Dominique Leblanc
- Sales is split into SMB (accounts < $50K ARR) and Enterprise (accounts ≥ $50K ARR).
- Enterprise team lead: **Marcus Webb**; current quota: $18M for FY2026.
- SMB team lead: **Tanya Osei**; current quota: $6M for FY2026.
- Solutions Engineering lead: **Wei Zhang** — 8 solutions engineers supporting pre-sales.

### 2.4 Marketing
- **VP of Marketing:** Rachel Huang
- Teams: Demand Generation, Content & Brand, and Product Marketing.
- Product Marketing Manager for Enterprise: **Sanjay Iyer**.

### 2.5 People Operations (HR)
- **Head of People Operations:** Claudia Torres
- Manages recruiting, total rewards, L&D, and DEI initiatives.
- Current open requisitions: 14 (as of March 2026).

### 2.6 Finance & Legal
- **CFO:** Priya Nambiar (also oversees Legal).
- General Counsel: **Derek Shin** (in-house since 2023; previously at Cooley LLP).
- Finance business partners are embedded in Engineering, Sales, and G&A.

### 2.7 Customer Success
- **VP of Customer Success:** Amara Diallo
- 22 Customer Success Managers (CSMs) organized by vertical (FinServ, Healthcare, Logistics).
- Average customer portfolio per CSM: ~8 accounts.
- Net Revenue Retention (NRR) target for FY2026: 118%.

---

## 3. Products

### 3.1 NovaPilot — AI Workflow Automation Platform
NovaPilot is NovaAI's flagship product, launched in **Q3 2022**. It allows enterprise customers to build, deploy, and monitor AI-powered workflow automations without writing code. NovaPilot integrates with over 120 third-party tools including Salesforce, SAP, and ServiceNow.

**Tiers and pricing (billed annually):**
| Tier | Price | Included |
|------|-------|----------|
| Starter | $2,500/month | Up to 10 users, 50K automation runs/mo |
| Growth | $8,000/month | Up to 50 users, 500K automation runs/mo, audit logs |
| Enterprise | Custom (typically $30K–$120K/mo) | Unlimited users, dedicated CSM, SLA 99.95% |

Key features: drag-and-drop workflow builder, real-time monitoring dashboard, role-based access control (RBAC), and a REST API with webhook support. The Enterprise tier includes **private cloud deployment** (AWS or Azure) and **HIPAA-compliant data handling**.

PM owner: **Fatima Al-Rashid**

---

### 3.2 NovaLens — Predictive Analytics Engine
NovaLens was launched in **Q1 2023** as a standalone analytics product. It ingests structured data from customer databases or data warehouses and surfaces predictive insights using pre-trained industry models that customers can fine-tune.

**Tiers and pricing (billed annually):**
| Tier | Price | Included |
|------|-------|----------|
| Professional | $4,000/month | 5 users, 3 data connectors, monthly model refresh |
| Business | $12,000/month | 20 users, 15 connectors, weekly refresh, custom dashboards |
| Enterprise | Custom | Unlimited connectors, daily refresh, bring-your-own-model support |

NovaLens currently supports connectors for **Snowflake, Databricks, BigQuery, Redshift, and PostgreSQL**. The product roadmap (H2 2026) includes native **real-time streaming** support via Kafka.

PM owner: **James Okafor**

---

### 3.3 NovaGuard — AI Security & Compliance Monitor
NovaGuard launched in **Q4 2023** and is NovaAI's newest generally available product. It monitors AI model outputs in production for bias, hallucination risk, and policy violations, and generates audit-ready compliance reports.

**Tiers and pricing (billed annually):**
| Tier | Price | Included |
|------|-------|----------|
| Standard | $3,000/month | 3 monitored models, weekly reports, email alerts |
| Advanced | $9,500/month | 20 models, daily reports, Slack/PagerDuty integration |
| Enterprise | Custom | Unlimited models, real-time alerting, custom policy rules |

NovaGuard is the fastest-growing product by new logo count (38 new customers in Q4 2025). It received a **SOC 2 Type II** certification in January 2026. The Standard and Advanced tiers are self-serve via the product portal; Enterprise requires a sales-assisted motion.

PM owner: **James Okafor** (dual ownership with NovaLens; a dedicated PM hire is planned for Q2 2026)

---

### 3.4 NovaBridge — Data Connector SDK (Developer Tool)
NovaBridge is a free, open-source SDK released in **March 2024** under the Apache 2.0 license. It enables developers to build custom connectors between NovaAI products and proprietary data sources. NovaBridge has **4,200+ GitHub stars** and an active community of ~800 contributors as of March 2026.

While NovaBridge itself is free, NovaAI offers a **NovaBridge Pro** support subscription at **$1,200/month**, which includes SLA-backed support, private forums, and quarterly architecture reviews with the ML Systems team. There are currently **47 NovaBridge Pro subscribers**.

NovaBridge is maintained by the Platform Infrastructure team under Raj Mehta, with developer relations managed by **Priya Osei** (Developer Relations Lead, reports to VP of Marketing).

---

## 4. Key Employees

| Name | Title | Department | Office | Start Date |
|------|-------|------------|--------|------------|
| Dr. Mara Chen | CEO & Co-founder | Executive | San Francisco | Jan 2021 |
| Lucas Ferreira | CTO & Co-founder | Engineering | San Francisco | Jan 2021 |
| Priya Nambiar | CFO | Finance | San Francisco | Apr 2022 |
| James Whitfield | COO | Operations | San Francisco | Feb 2023 |
| Sofia Andersson | VP of Engineering | Engineering | San Francisco | Aug 2021 |
| Oliver Haines | VP of Product | Product | London | Mar 2022 |
| Dominique Leblanc | CRO | Sales | San Francisco | Jun 2022 |
| Amara Diallo | VP of Customer Success | CS | San Francisco | Sep 2022 |
| Rachel Huang | VP of Marketing | Marketing | San Francisco | Jan 2023 |
| Claudia Torres | Head of People Ops | People | San Francisco | Nov 2021 |
| Derek Shin | General Counsel | Legal | San Francisco | May 2023 |
| Dr. Yuki Tanaka | ML Systems Lead | Engineering | San Francisco | Oct 2021 |
| Raj Mehta | Platform Infra Lead | Engineering | Singapore | Jul 2022 |
| Lena Park | Head of Design | Product | London | Jun 2022 |
| Marcus Webb | Enterprise Sales Lead | Sales | New York* | Jan 2024 |
| Ananya Bose | Security Eng. Lead | Engineering | San Francisco | Mar 2022 |
| Fatima Al-Rashid | PM – NovaPilot | Product | San Francisco | May 2022 |
| James Okafor | PM – NovaLens & NovaGuard | Product | London | Sep 2023 |
| Wei Zhang | Solutions Eng. Lead | Sales | San Francisco | Apr 2023 |
| Priya Osei | Developer Relations Lead | Marketing | San Francisco | Aug 2024 |

*Marcus Webb works remotely from New York under an approved remote work agreement.

---

## 5. Internal Policies

### 5.1 Paid Time Off (PTO)
NovaAI operates on a **flexible/unlimited PTO** policy for all full-time employees. There is no accrual or cap, but employees are expected to take a **minimum of 15 days per calendar year**. PTO requests must be submitted in Workday at least **5 business days in advance** for absences of 3+ days. Manager approval is required for all PTO.

In addition to flexible PTO, the company observes **11 paid company holidays** per year. Employees in London and Singapore follow their local public holiday calendars, adjusted to match the 11-day total.

**Sick leave** is tracked separately and does not count against PTO. Employees should notify their manager by 9:00 AM on any day they are unable to work due to illness.

### 5.2 Remote Work Policy
NovaAI is a **hybrid-first** company. The default expectation is that SF-based employees come into the office **Tuesday and Thursday** each week. London and Singapore offices follow the same two-day in-office requirement on their local Tuesday/Thursday.

Employees may request **fully remote** arrangements through People Ops. Approval requires sign-off from the employee's direct manager, the department VP, and Claudia Torres. Fully remote arrangements are reviewed annually.

Employees working remotely outside their home country for more than **14 consecutive days** must notify People Ops in advance due to tax and employment law implications.

Equipment stipend for home office setup: **$1,500 one-time** upon hire, refreshable every 3 years.

### 5.3 Expense Reimbursement
All business expenses must be submitted through **Expensify** within **30 days** of the expense date. Receipts are required for any expense over **$25**.

**Pre-approved spending categories (no manager approval needed up to the limit):**
| Category | Limit |
|----------|-------|
| Team meals (host of 2+ people) | $75/person |
| Travel: ground transport | $150/day |
| Travel: hotel | $350/night (SF/NYC/London); $250/night (other) |
| Conference registration | $1,500/event |
| Books & learning materials | $100/month |

Airfare must be booked through **TravelPerk** (NovaAI's travel management platform). Economy class is required for flights under 6 hours; business class is permitted for flights of 6 hours or more with VP+ approval.

Expenses over $5,000 require CFO (Priya Nambiar) approval regardless of category.

### 5.4 Performance Review Cycle
NovaAI runs a **bi-annual performance review cycle**:
- **Mid-year review:** Calibration in **July**, results communicated to employees by **July 31**.
- **Annual review:** Calibration in **January**, results communicated by **January 31**. Annual reviews are tied to compensation adjustments and promotion decisions.

The review process uses a **5-point rating scale**: Exceptional, Exceeds Expectations, Meets Expectations, Developing, and Unsatisfactory. Ratings of Developing or below trigger a formal **Performance Improvement Plan (PIP)** within 30 days, managed jointly by the employee's manager and People Ops.

Employees must have been in their role for at least **90 days** to receive a formal rating. New hires within 90 days of a review cycle receive a "Too New to Rate" designation.

Promotions are decided during the **annual review cycle only**, except for off-cycle promotions at VP+ level, which require CEO approval.

### 5.5 Learning & Development
Each employee receives an **annual L&D budget of $2,500**, refreshed on January 1. This covers conferences, online courses, certifications, and workshops. Unused budget does not roll over. Requests are submitted via the L&D portal in Workday.

NovaAI also offers an internal **Tech Talk** series every other Thursday at 12 PM PT (streamed to all offices), where engineers and PMs present on projects, external research, or technical concepts.

---

## 6. Miscellaneous Policies

- **Equity:** All full-time employees receive stock option grants. Standard vesting is **4 years with a 1-year cliff**. Option details are managed through **Carta**.
- **Health benefits (US):** Medical, dental, and vision through **Anthem Blue Cross**. NovaAI covers **90% of employee premiums** and **70% of dependent premiums**.
- **401(k):** NovaAI matches **4% of salary** (dollar-for-dollar), with a 6-month eligibility waiting period.
- **Parental leave:** **16 weeks** fully paid for all parents (birthing and non-birthing), regardless of gender.
- **Code of Conduct violations** should be reported to People Ops or via the anonymous hotline managed by **EthicsPoint** (accessible at the internal portal).
- **On-call rotations** for platform reliability are managed by the Platform Infrastructure team. On-call compensation is an additional **$300/week** while on-call.

---

*For questions about this document, contact People Operations at people@nova-ai.internal or ping the #people-ops Slack channel.*
