import json
import random
from pathlib import Path
from typing import Any, Dict, List

BASE_DIR = Path(__file__).parent
ROLES_PATH = BASE_DIR / "roles.json"
OUT_PATH = BASE_DIR / "candidates.json"

# Domains and keyword cues for simple clustering.
DOMAIN_KEYWORDS = {
    "ml": ["ml", "model", "training", "inference", "multimodal", "rl", "alignment", "safety", "vision", "audio"],
    "infra": ["infrastructure", "platform", "sre", "storage", "network", "datacenter", "traffic", "scaling", "kubernetes"],
    "backend": ["backend", "distributed", "api", "microservice", "rust", "c++", "grok", "enterprise"],
    "data": ["data", "analytics", "etl", "pipeline", "experiment", "product analytics"],
    "security": ["security", "detection", "response", "appsec", "infra security"],
    "research": ["research", "science", "evaluation", "epistemics"],
    "product": ["product manager", "pm", "designer", "design", "growth"],
}

LOCATIONS_FALLBACK = ["Palo Alto, CA", "San Francisco, CA", "Seattle, WA", "London, UK", "New York, NY", "Remote"]

FIRST_NAMES = [
    "Ava", "Noah", "Liam", "Mia", "Ethan", "Sofia", "Leo", "Isla", "Maya", "Jude",
    "Iris", "Caleb", "Aria", "Nico", "Zara", "Kai", "Elena", "Mason", "Nova", "Elias",
    "Priya", "Ravi", "Sanaa", "Mateo", "Yara", "Samir", "Anya", "Dante", "Lucia", "Rowan"
]

LAST_NAMES = [
    "Chen", "Patel", "Garcia", "Nguyen", "Brown", "Davis", "Khan", "Yamamoto", "Kim",
    "Singh", "Lopez", "Martinez", "Silva", "Rossi", "Ivanov", "Williams", "Jones",
    "Rahman", "Hernandez", "Smith", "Kaur", "Ali", "Sato", "Dubois", "Kowalski",
    "Costa", "Meier", "Andersson", "Hughes", "Murphy"
]


def load_roles() -> List[Dict[str, Any]]:
    with ROLES_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def classify_domain(title: str) -> str:
    t = title.lower()
    for domain, keywords in DOMAIN_KEYWORDS.items():
        if any(k in t for k in keywords):
            return domain
    return "ml"  # default bias toward ML/eng


def pick_location(role: Dict[str, Any]) -> str:
    loc = role.get("location")
    if loc:
        return loc
    return random.choice(LOCATIONS_FALLBACK)


def synth_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def synth_summary(role_title: str, domain: str) -> str:
    templates = {
        "ml": "ML engineer shipping LLM/vision models with tight feedback loops between training and inference.",
        "infra": "Infra engineer who hardens and scales distributed systems for low-latency model serving.",
        "backend": "Backend engineer who owns APIs, services, and data flows that power product experiences.",
        "data": "Data/analytics engineer who builds pipelines, runs experiments, and closes the loop with product.",
        "security": "Security engineer focused on detection/response and hardening critical paths.",
        "research": "Applied scientist bridging research ideas into production-grade evaluations and models.",
        "product": "Product/design leader who pairs tightly with engineering to ship fast and iterate.",
    }
    base = templates.get(domain, templates["ml"])
    return f"{base} Interested in the {role_title} opening."


def synth_skills(domain: str) -> List[str]:
    base = {
        "ml": ["PyTorch", "JAX", "Transformers", "LoRA", "Eval pipelines", "RLHF/RLAIF", "CUDA basics"],
        "infra": ["Kubernetes", "Service mesh", "gRPC", "Observability", "Caching", "Databases", "Linux perf"],
        "backend": ["Python", "Go", "Rust", "gRPC/REST", "PostgreSQL", "Redis", "Tracing"],
        "data": ["Python", "SQL", "Spark", "dbt", "Airflow", "Experimentation", "Dashboards"],
        "security": ["Detection", "Incident response", "SIEM", "Cloud hardening", "AppSec review"],
        "research": ["PyTorch", "Eval design", "Prompting", "Data curation", "Benchmarking"],
        "product": ["Product discovery", "Spec writing", "User research", "A/B testing", "Design systems"],
    }
    extras = {
        "ml": ["NLP", "Vision", "Multimodal", "Quantization", "Inference optimization"],
        "infra": ["Networking", "Storage", "Scheduling", "Reliability", "Throughput/latency tuning"],
        "backend": ["API design", "Microservices", "Performance tuning"],
        "data": ["Causal inference", "Experiment design", "Metrics"],
        "security": ["Threat modeling", "Forensics"],
        "research": ["Paper reproduction", "Rapid prototyping"],
        "product": ["Roadmapping", "Stakeholder comms"],
    }
    skills = base.get(domain, []) + extras.get(domain, [])
    random.shuffle(skills)
    return skills[:8]


def synth_projects(domain: str) -> List[str]:
    options = {
        "ml": [
            "Shipped LLM-based feature ranking with online evals.",
            "Built distillation pipeline to shrink models for edge inference.",
            "Owned safety/guardrail prompts and red-teaming loops."
        ],
        "infra": [
            "Scaled model-serving stack to double QPS at lower p99 latency.",
            "Implemented autoscaling and circuit breakers for inference services.",
            "Migrated observability stack to cut MTTR on incidents."
        ],
        "backend": [
            "Designed high-QPS API for conversational features.",
            "Reduced tail latencies via async and better caching.",
            "Refactored monolith components into typed services."
        ],
        "data": [
            "Built ELT + experiment readouts for product launches.",
            "Implemented metrics guardrails and anomaly detection.",
            "Partnered with PMs to define north-star metrics."
        ],
        "security": [
            "Built detections for auth anomalies and lateral movement.",
            "Ran incident postmortems and hardened auth flows.",
            "Partnered with eng to threat-model new services."
        ],
        "research": [
            "Designed evals for reasoning prompts and long-context tasks.",
            "Prototyped retrieval-augmented generation pipelines.",
            "Benchmarked multimodal models on custom tasks."
        ],
        "product": [
            "Drove 0→1 launch with tight design/eng loop.",
            "Ran multi-variant experiments to improve activation.",
            "Built fast spec → prototype → feedback workflow."
        ],
    }
    picks = options.get(domain, options["ml"])
    random.shuffle(picks)
    return picks[:3]


def synth_experience(domain: str) -> List[Dict[str, Any]]:
    companies = ["Aurora Labs", "Nimbus Systems", "Vector Forge", "Helix AI", "Northstar Compute", "Cinder Cloud"]
    titles = {
        "ml": ["ML Engineer", "Applied Scientist", "ML Infra Engineer"],
        "infra": ["Infrastructure Engineer", "SRE", "Platform Engineer"],
        "backend": ["Backend Engineer", "Software Engineer"],
        "data": ["Data Engineer", "Analytics Engineer", "Data Scientist"],
        "security": ["Security Engineer", "Detection & Response Engineer"],
        "research": ["Research Engineer", "Applied Researcher"],
        "product": ["Product Manager", "Product Designer"],
    }
    bullets = synth_projects(domain)
    return [
        {
            "company": random.choice(companies),
            "title": random.choice(titles.get(domain, ["Software Engineer"])),
            "tenure": "1.5-3 years",
            "highlights": bullets
        },
        {
            "company": random.choice(companies),
            "title": random.choice(titles.get(domain, ["Software Engineer"])),
            "tenure": "1-2 years",
            "highlights": bullets[:2]
        },
    ]


def build_candidate(role: Dict[str, Any], cid: int) -> Dict[str, Any]:
    domain = classify_domain(role.get("title", ""))
    name = synth_name()
    location = pick_location(role)
    skills = synth_skills(domain)
    return {
        "id": f"cand-{cid:03d}",
        "name": name,
        "title": role.get("title"),
        "target_role_url": role.get("absolute_url"),
        "location": location,
        "timezone": "Aligned with role location or PST" if "Palo Alto" in location or "San Francisco" in location else "Flexible/Remote",
        "seniority": random.choice(["mid", "senior", "staff"]),
        "domain": domain,
        "summary": synth_summary(role.get("title", ""), domain),
        "skills_core": skills,
        "skills_secondary": skills[-3:],
        "projects": synth_projects(domain),
        "recent_experience": synth_experience(domain),
        "education": random.choice([
            "BS Computer Science",
            "MS Computer Science",
            "BS Electrical Engineering",
            "MS Data Science"
        ]),
        "openness": random.choice([
            "Open to Palo Alto/SF onsite; remote OK for right team",
            "Prefers remote; can travel quarterly",
            "Open to relocation to Bay Area; remote hybrid OK"
        ]),
        "job_description_excerpt": role.get("content_text", "")[:1200].strip()
    }


def main(count: int = 25) -> None:
    random.seed(42)
    roles = load_roles()
    # Filter to roles with content and tilt toward eng/ML.
    prioritized = [
        r for r in roles
        if r.get("content_text")
        and classify_domain(r.get("title", "")) in {"ml", "infra", "backend", "data", "security", "research"}
    ]
    if len(prioritized) < count:
        prioritized = roles

    sampled = prioritized[:count]
    candidates = [build_candidate(role, idx + 1) for idx, role in enumerate(sampled)]

    with OUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(candidates, fh, ensure_ascii=False, indent=2)
    print(f"Wrote {len(candidates)} candidates to {OUT_PATH}")


if __name__ == "__main__":
    main()

