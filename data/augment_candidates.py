"""
Augment synthetic candidate profiles with DIVERSE, UNIQUE details.

This version generates candidates with:
- Unique project descriptions with specific metrics/technologies
- Edge cases: career changers, location mismatches, seniority gaps
- Quirks and unusual backgrounds that require genuine personalization
- NO pre-stated interest in the role (makes the task harder)

Usage:
  python data/augment_candidates.py --roles roles.json --count 25
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES = BASE_DIR / "data" / "candidates.json"
DEFAULT_ROLES = BASE_DIR / "data" / "roles.json"
DEFAULT_FORMATTED = BASE_DIR / "data" / "candidates_formatted.jsonl"

DOMAIN_KEYWORDS = {
    "ml": ["ml", "model", "training", "inference", "multimodal", "rl", "alignment", "safety", "vision", "audio", "tutor"],
    "infra": ["infrastructure", "platform", "sre", "storage", "network", "datacenter", "traffic", "scaling", "kubernetes"],
    "backend": ["backend", "distributed", "api", "microservice", "rust", "c++", "grok", "enterprise"],
    "data": ["data", "analytics", "etl", "pipeline", "experiment", "product analytics"],
    "security": ["security", "detection", "response", "appsec", "infra security"],
    "research": ["research", "science", "evaluation", "epistemics"],
    "product": ["product manager", "pm", "designer", "design", "growth"],
}

# Expanded location pools for creating mismatches
LOCATIONS_MATCH = [
    "Palo Alto, CA",
    "San Francisco, CA",
    "Remote",
]

LOCATIONS_MISMATCH = [
    "Berlin, Germany",
    "Tokyo, Japan",
    "Sydney, Australia",
    "Toronto, Canada",
    "Singapore",
    "Sao Paulo, Brazil",
    "Stockholm, Sweden",
    "Tel Aviv, Israel",
    "Bangalore, India",
    "Amsterdam, Netherlands",
]

FIRST_NAMES = [
    "Ava", "Noah", "Liam", "Mia", "Ethan", "Sofia", "Leo", "Isla", "Maya", "Jude",
    "Iris", "Caleb", "Aria", "Nico", "Zara", "Kai", "Elena", "Mason", "Nova", "Elias",
    "Priya", "Ravi", "Sanaa", "Mateo", "Yara", "Samir", "Anya", "Dante", "Lucia", "Rowan",
    "Felix", "Camila", "Omar", "Selene", "Ada", "Hugo", "Imani", "Jonas", "Tara", "Zane",
    "Wei", "Aisha", "Dmitri", "Kenji", "Fatima", "Olga", "Henrik", "Yuki", "Rashid", "Linnea",
]

LAST_NAMES = [
    "Chen", "Patel", "Garcia", "Nguyen", "Brown", "Davis", "Khan", "Yamamoto", "Kim",
    "Singh", "Lopez", "Martinez", "Silva", "Rossi", "Ivanov", "Williams", "Jones",
    "Rahman", "Hernandez", "Smith", "Kaur", "Ali", "Sato", "Dubois", "Kowalski",
    "Costa", "Meier", "Andersson", "Hughes", "Murphy", "Zhang", "Okonkwo", "Petrov",
]

# Real-ish company names for variety
COMPANIES = [
    "Stripe", "Anthropic", "DeepMind", "Meta AI", "Google Brain", "OpenAI", "Cohere",
    "Databricks", "Snowflake", "Scale AI", "Weights & Biases", "Hugging Face",
    "Cruise", "Waymo", "Tesla Autopilot", "Aurora Innovation", "Nuro",
    "Figma", "Notion", "Linear", "Vercel", "Supabase", "PlanetScale",
    "Cloudflare", "Fastly", "Akamai", "HashiCorp", "Confluent", "MongoDB",
    "Palantir", "Datadog", "Splunk", "Elastic", "New Relic", "Grafana Labs",
    "Jane Street", "Two Sigma", "Citadel", "HRT", "Jump Trading",
    "Coinbase", "Ripple", "Circle", "Chainalysis", "Fireblocks",
    "Instacart", "DoorDash", "Uber Eats", "Grubhub",
]

# Specific metric templates for unique project descriptions
PROJECT_METRICS = [
    "reduced latency from {high}ms to {low}ms",
    "improved throughput by {pct}%",
    "cut inference costs by ${amount}K/month",
    "scaled to {num}M daily active users",
    "reduced model size by {pct}% while maintaining accuracy",
    "achieved {pct}% accuracy on {benchmark}",
    "decreased training time from {high} hours to {low} hours",
    "processed {num}B tokens/day",
    "served {num}K QPS at p99 < {low}ms",
]

# Unique project templates with slots for specifics
PROJECT_TEMPLATES = {
    "ml": [
        "Led migration from {old_framework} to {new_framework}, {metric}",
        "Built {model_type} fine-tuning pipeline for {use_case}; {metric}",
        "Designed custom {technique} for {problem}, achieving {metric}",
        "Shipped {feature} using {tech_stack}; now handles {scale}",
        "Owned {component} system end-to-end, from training to serving",
        "Red-teamed {model_name} for {safety_area}; found {num} critical issues",
        "Created internal {tool_type} that {outcome}",
        "Collaborated with {team} to build {product}; {impact}",
    ],
    "infra": [
        "Migrated {service} to {cloud/tech}, {metric}",
        "Built {system_type} from scratch; now handles {scale}",
        "Debugged {incident_type} affecting {scope}; {outcome}",
        "Automated {process} using {tools}, saving {time}/week",
        "Designed {architecture} for {requirement}",
        "Led on-call rotation for {num} services; achieved {uptime}% uptime",
    ],
    "backend": [
        "Rewrote {service} in {language}, {metric}",
        "Designed {api_type} API serving {scale}",
        "Built {feature} that {outcome}",
        "Migrated {data_amount} from {old_system} to {new_system}",
        "Optimized {component} reducing {resource} usage by {pct}%",
    ],
}

# Quirks and unusual background elements
QUIRKS = [
    "former {previous_career} who transitioned to tech",
    "dropped out of {degree} to join {company}",
    "self-taught programmer, no CS degree",
    "published {num} papers on {topic}",
    "maintainer of {oss_project} ({stars}+ GitHub stars)",
    "previously founded {startup_type} startup (acquired/failed)",
    "worked remotely from {num} countries in the past year",
    "on a career break for {duration} (traveling/family/health)",
    "dual background in {field1} and {field2}",
    "speaker at {conference}",
    "wrote {blog_post} that got {num}K views",
]

PREVIOUS_CAREERS = ["teacher", "musician", "physicist", "financial analyst", "doctor", "lawyer", "architect", "chef"]
OSS_PROJECTS = ["vLLM", "LangChain", "FastAPI", "Pydantic", "FAISS", "Transformers", "PyTorch Lightning", "MLflow"]
CONFERENCES = ["NeurIPS", "ICML", "KDD", "Strange Loop", "QCon", "PyCon", "KubeCon", "re:Invent"]
TOPICS = ["attention mechanisms", "distributed training", "model compression", "RLHF", "code generation", "multimodal learning"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def classify_domain(title: str) -> str:
    t = title.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in t for k in keywords):
            return domain
    return "ml"


def synth_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def synth_unique_project(domain: str) -> str:
    """Generate a unique project description with specific details."""
    templates = PROJECT_TEMPLATES.get(domain, PROJECT_TEMPLATES["ml"])
    template = random.choice(templates)
    
    # Fill in template slots with random specifics
    substitutions = {
        "{old_framework}": random.choice(["TensorFlow", "Keras", "custom C++", "legacy PyTorch 1.x"]),
        "{new_framework}": random.choice(["JAX/Flax", "PyTorch 2.0", "MLX", "Triton"]),
        "{model_type}": random.choice(["LoRA", "QLoRA", "full fine-tuning", "prefix-tuning", "PEFT"]),
        "{use_case}": random.choice(["customer support", "code completion", "document summarization", "search ranking", "content moderation"]),
        "{technique}": random.choice(["attention mechanism", "loss function", "sampling strategy", "data augmentation", "distillation method"]),
        "{problem}": random.choice(["long-context handling", "multilingual support", "low-resource languages", "domain adaptation"]),
        "{feature}": random.choice(["real-time inference", "batch prediction", "A/B testing framework", "feature store", "model registry"]),
        "{tech_stack}": random.choice(["Ray Serve + vLLM", "TensorRT + Triton", "custom CUDA kernels", "ONNX Runtime"]),
        "{scale}": random.choice(["10K QPS", "100M daily requests", "500GB/day throughput", "1M concurrent users"]),
        "{component}": random.choice(["embedding", "tokenizer", "inference", "training", "evaluation"]),
        "{model_name}": random.choice(["Llama-3", "GPT-4", "Claude", "Gemini", "internal 70B model"]),
        "{safety_area}": random.choice(["jailbreaks", "data leakage", "prompt injection", "harmful content"]),
        "{num}": str(random.randint(3, 50)),
        "{tool_type}": random.choice(["eval harness", "debugging tool", "annotation platform", "monitoring dashboard"]),
        "{outcome}": random.choice(["reduced iteration time by 60%", "caught 3 production bugs pre-launch", "used by 50+ engineers daily"]),
        "{team}": random.choice(["research", "product", "safety", "infra"]),
        "{product}": random.choice(["chat feature", "search ranking", "content recommendation", "fraud detection"]),
        "{impact}": random.choice(["shipped to 10M users", "increased engagement 15%", "reduced support tickets 40%"]),
        "{metric}": random.choice(PROJECT_METRICS).format(
            high=random.randint(200, 500),
            low=random.randint(10, 50),
            pct=random.randint(20, 80),
            amount=random.randint(50, 500),
            num=random.randint(1, 100),
            benchmark=random.choice(["MMLU", "HumanEval", "GSM8K", "internal benchmark"]),
        ),
        "{service}": random.choice(["auth service", "notification system", "payment processor", "search index"]),
        "{cloud/tech}": random.choice(["Kubernetes", "serverless", "edge compute", "multi-region"]),
        "{system_type}": random.choice(["job scheduler", "rate limiter", "cache layer", "message queue"]),
        "{incident_type}": random.choice(["cascading failure", "memory leak", "network partition", "thundering herd"]),
        "{scope}": random.choice(["50% of traffic", "all EU users", "mobile clients", "enterprise customers"]),
        "{process}": random.choice(["deployments", "rollbacks", "capacity planning", "incident response"]),
        "{tools}": random.choice(["Terraform + Pulumi", "custom CLI", "GitHub Actions", "Argo CD"]),
        "{time}": random.choice(["10 hours", "2 days", "40 engineer-hours"]),
        "{architecture}": random.choice(["multi-tenant", "event-driven", "CQRS", "sharded"]),
        "{requirement}": random.choice(["100ms p99 latency", "99.99% availability", "GDPR compliance", "SOC2 audit"]),
        "{uptime}": str(random.uniform(99.9, 99.99))[:5],
        "{language}": random.choice(["Rust", "Go", "async Python", "C++"]),
        "{api_type}": random.choice(["GraphQL", "gRPC", "REST", "streaming"]),
        "{data_amount}": random.choice(["50TB", "10B rows", "5 years of history"]),
        "{old_system}": random.choice(["MySQL", "MongoDB", "Cassandra", "Redis"]),
        "{new_system}": random.choice(["PostgreSQL", "CockroachDB", "TiDB", "DynamoDB"]),
        "{resource}": random.choice(["memory", "CPU", "network", "storage"]),
        "{pct}": str(random.randint(20, 70)),
    }
    
    result = template
    for key, value in substitutions.items():
        result = result.replace(key, value)
    return result


def synth_quirk() -> str | None:
    """Generate a unique background quirk (30% chance)."""
    if random.random() > 0.3:
        return None
    
    quirk = random.choice(QUIRKS)
    substitutions = {
        "{previous_career}": random.choice(PREVIOUS_CAREERS),
        "{degree}": random.choice(["PhD", "MS", "undergrad"]),
        "{company}": random.choice(COMPANIES[:10]),
        "{num}": str(random.randint(2, 15)),
        "{topic}": random.choice(TOPICS),
        "{oss_project}": random.choice(OSS_PROJECTS),
        "{stars}": str(random.randint(1, 50)) + "K",
        "{startup_type}": random.choice(["AI", "SaaS", "fintech", "devtools"]),
        "{duration}": random.choice(["6 months", "1 year", "18 months"]),
        "{field1}": random.choice(["ML", "economics", "physics", "neuroscience"]),
        "{field2}": random.choice(["product management", "systems engineering", "data science"]),
        "{conference}": random.choice(CONFERENCES),
        "{blog_post}": random.choice(['"Why I Left Big Tech"', '"Scaling LLMs on a Budget"', '"The Death of Fine-Tuning"']),
    }
    
    result = quirk
    for key, value in substitutions.items():
        result = result.replace(key, value)
    return result


def synth_summary(domain: str, quirk: str | None) -> str:
    """Generate summary WITHOUT mentioning specific role interest."""
    templates = {
        "ml": [
            "ML engineer focused on production LLM systems.",
            "Building inference infrastructure for large language models.",
            "Applied ML engineer with focus on model optimization and deployment.",
            "Working on RLHF and model alignment.",
            "Specializing in multimodal models and vision-language systems.",
        ],
        "infra": [
            "Infrastructure engineer scaling distributed systems.",
            "SRE focused on high-availability model serving.",
            "Platform engineer building ML infrastructure.",
            "Specializing in GPU cluster management and networking.",
        ],
        "backend": [
            "Backend engineer building high-throughput APIs.",
            "Full-stack with backend focus, shipping user-facing products.",
            "Systems engineer with distributed systems expertise.",
        ],
        "data": [
            "Data engineer building real-time analytics pipelines.",
            "Analytics engineer focused on experimentation and causal inference.",
        ],
        "security": [
            "Security engineer focused on detection and response.",
            "AppSec engineer hardening production systems.",
        ],
        "research": [
            "Applied researcher bridging research and production.",
            "Research engineer focused on evaluation and benchmarking.",
        ],
        "product": [
            "Product manager with technical background.",
            "Design-focused PM shipping developer tools.",
        ],
    }
    
    base = random.choice(templates.get(domain, templates["ml"]))
    if quirk:
        return f"{base} {quirk.capitalize()}."
    return base


def synth_skills(domain: str) -> List[str]:
    """Generate skills with some noise/variety."""
    base = {
        "ml": ["PyTorch", "JAX", "Transformers", "LoRA", "vLLM", "RLHF", "CUDA", "Triton", "TensorRT", "MLflow", "Ray", "DeepSpeed"],
        "infra": ["Kubernetes", "Terraform", "gRPC", "Prometheus", "Redis", "PostgreSQL", "Linux", "TCP/IP", "BGP", "RDMA"],
        "backend": ["Python", "Go", "Rust", "TypeScript", "PostgreSQL", "Redis", "Kafka", "gRPC", "GraphQL"],
        "data": ["Python", "SQL", "Spark", "dbt", "Airflow", "Flink", "BigQuery", "Snowflake"],
        "security": ["SIEM", "Splunk", "AWS Security", "Threat Modeling", "Incident Response", "Forensics"],
        "research": ["PyTorch", "JAX", "Eval Design", "Statistical Analysis", "Paper Writing"],
        "product": ["Product Strategy", "User Research", "SQL", "A/B Testing", "Figma"],
    }
    
    skills = base.get(domain, base["ml"]).copy()
    random.shuffle(skills)
    return skills[:random.randint(5, 8)]


def synth_experience(domain: str) -> List[Dict[str, Any]]:
    """Generate experience with UNIQUE companies and specific projects."""
    titles = {
        "ml": ["ML Engineer", "Applied Scientist", "ML Platform Engineer", "Research Engineer", "AI Engineer"],
        "infra": ["Infrastructure Engineer", "SRE", "Platform Engineer", "DevOps Engineer", "Systems Engineer"],
        "backend": ["Backend Engineer", "Software Engineer", "Senior SWE", "Staff Engineer"],
        "data": ["Data Engineer", "Analytics Engineer", "Data Scientist"],
        "security": ["Security Engineer", "Detection Engineer", "AppSec Engineer"],
        "research": ["Research Engineer", "Research Scientist", "Applied Researcher"],
        "product": ["Product Manager", "Senior PM", "Product Designer"],
    }
    
    # Pick 2 different companies
    companies = random.sample(COMPANIES, 2)
    
    return [
        {
            "company": companies[0],
            "title": random.choice(titles.get(domain, ["Software Engineer"])),
            "tenure": random.choice(["1.5 years", "2 years", "2.5 years", "3 years"]),
            "highlights": [synth_unique_project(domain) for _ in range(random.randint(2, 3))],
        },
        {
            "company": companies[1],
            "title": random.choice(titles.get(domain, ["Software Engineer"])),
            "tenure": random.choice(["1 year", "1.5 years", "2 years"]),
            "highlights": [synth_unique_project(domain) for _ in range(random.randint(1, 2))],
        },
    ]


def pick_location(role: Dict[str, Any], edge_case: bool) -> tuple[str, str]:
    """Return (location, timezone). edge_case=True for location mismatches."""
    role_loc = role.get("location") or ""
    
    if edge_case:
        # 30% chance of significant location mismatch
        loc = random.choice(LOCATIONS_MISMATCH)
        tz = "Not aligned with US timezones"
        return loc, tz
    
    # Otherwise roughly match or be remote-flexible
    if "Remote" in role_loc or random.random() > 0.5:
        return "Remote", "Flexible (can align to PST)"
    
    if "Palo Alto" in role_loc or "San Francisco" in role_loc:
        return random.choice(LOCATIONS_MATCH[:2]), "PST"
    
    return role_loc or random.choice(LOCATIONS_MATCH), "Flexible"


def pick_seniority(role: Dict[str, Any], edge_case: bool) -> str:
    """Pick seniority, potentially mismatched for edge cases."""
    if edge_case:
        # Mismatch: senior candidate for junior role or vice versa
        return random.choice(["director", "principal", "junior", "new grad"])
    return random.choice(["mid", "senior", "staff"])


def build_candidate(role: Dict[str, Any], cid: int, edge_case_pct: float = 0.25) -> Dict[str, Any]:
    """Build a candidate with unique details. edge_case_pct controls adversarial examples."""
    domain = classify_domain(role.get("title", ""))
    is_edge_case = random.random() < edge_case_pct
    
    # Sometimes create domain mismatch (e.g., backend engineer for ML role)
    if is_edge_case and random.random() > 0.5:
        domain = random.choice(["ml", "infra", "backend", "data"])
    
    name = synth_name()
    location, timezone = pick_location(role, is_edge_case)
    seniority = pick_seniority(role, is_edge_case)
    quirk = synth_quirk()
    skills = synth_skills(domain)
    
    return {
        "id": f"cand-{cid:03d}",
        "name": name,
        "title": role.get("title"),
        "target_role_url": role.get("absolute_url"),
        "location": location,
        "timezone": timezone,
        "seniority": seniority,
        "domain": domain,
        "summary": synth_summary(domain, quirk),
        "quirk": quirk,  # Store for reference
        "is_edge_case": is_edge_case,
        "skills_core": skills,
        "projects": [synth_unique_project(domain) for _ in range(3)],
        "recent_experience": synth_experience(domain),
        "education": random.choice([
            "BS Computer Science, Stanford",
            "MS Machine Learning, CMU",
            "PhD candidate (ABD), MIT",
            "Self-taught, bootcamp graduate",
            "BS Physics, converted to ML",
            "MS Data Science, Berkeley",
            "BS EE, Georgia Tech",
            "No degree, 8 years experience",
        ]),
        "openness": random.choice([
            "Actively looking",
            "Open to hearing about senior roles",
            "Not actively looking but always curious",
            "Happy where I am, but open to exceptional opportunities",
            "Exploring options after recent layoff",
            "Looking to relocate to Bay Area",
            "Remote-only, non-negotiable",
            "Open to contract or full-time",
        ]),
        "job_description_excerpt": (role.get("content_text") or role.get("content_html") or "")[:1200].strip(),
    }


def render_profile(c: dict) -> str:
    """Render candidate profile for prompt context."""
    parts = [
        f"Name: {c.get('name')}",
        f"Target role: {c.get('title')} ({c.get('target_role_url')})",
        f"Location: {c.get('location')} | Timezone: {c.get('timezone')}",
        f"Seniority: {c.get('seniority')} | Domain: {c.get('domain')}",
        f"Summary: {c.get('summary')}",
    ]
    
    skills = ", ".join(c.get("skills_core", []))
    if skills:
        parts.append(f"Skills: {skills}")
    
    projects = c.get("projects") or []
    if projects:
        parts.append("Notable projects:\n- " + "\n- ".join(projects))
    
    exp = c.get("recent_experience") or []
    if exp:
        bullets = []
        for role in exp:
            highlights = role.get("highlights") or []
            hl = "; ".join(highlights)
            bullets.append(f"{role.get('title')} @ {role.get('company')} ({role.get('tenure')}): {hl}")
        parts.append("Experience:\n- " + "\n- ".join(bullets))
    
    edu = c.get("education")
    if edu:
        parts.append(f"Education: {edu}")
    
    open_note = c.get("openness")
    if open_note:
        parts.append(f"Job search status: {open_note}")
    
    jd_excerpt = c.get("job_description_excerpt") or ""
    if jd_excerpt:
        parts.append("Job excerpt:\n" + jd_excerpt.strip())
    
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formatted", type=Path, default=DEFAULT_FORMATTED)
    parser.add_argument("--count", type=int, default=50, help="Number of candidates to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--edge-case-pct", type=float, default=0.25, help="Fraction of adversarial examples")
    parser.add_argument("--replace", action="store_true", help="Replace existing candidates instead of appending")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    
    roles = load_json(args.roles)
    
    if args.replace or not args.candidates.exists():
        candidates_existing = []
    else:
        candidates_existing = load_json(args.candidates)

    start_idx = len(candidates_existing) + 1
    
    # Prioritize roles with content
    prioritized = [r for r in roles if r.get("content_text")]
    if len(prioritized) < args.count:
        prioritized = roles
    
    # Sample roles, cycling if needed
    sampled = []
    for i in range(args.count):
        sampled.append(prioritized[i % len(prioritized)])

    new_candidates = [
        build_candidate(role, start_idx + i, edge_case_pct=args.edge_case_pct) 
        for i, role in enumerate(sampled)
    ]
    all_candidates = candidates_existing + new_candidates

    # Write updated candidates.json
    args.candidates.parent.mkdir(parents=True, exist_ok=True)
    with args.candidates.open("w", encoding="utf-8") as fh:
        json.dump(all_candidates, fh, ensure_ascii=False, indent=2)

    # Write formatted JSONL
    args.formatted.parent.mkdir(parents=True, exist_ok=True)
    with args.formatted.open("w", encoding="utf-8") as out:
        for c in all_candidates:
            profile_text = render_profile(c)
            record = {"messages": [{"role": "user", "content": profile_text}]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    edge_cases = sum(1 for c in new_candidates if c.get("is_edge_case"))
    print(f"Generated {len(new_candidates)} candidates ({edge_cases} edge cases). Total: {len(all_candidates)}")
    print(f"Wrote {args.candidates} and {args.formatted}")


if __name__ == "__main__":
    main()
