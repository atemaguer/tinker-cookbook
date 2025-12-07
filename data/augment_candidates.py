"""
Augment synthetic candidate profiles with DIVERSE, ROLE-APPROPRIATE details.

This version generates candidates whose backgrounds actually match (or intentionally
mismatch) the target roles. For tutor roles requiring PhDs/domain expertise, we
generate domain experts. For engineering roles, we generate engineers. etc.

Edge cases are candidates with partial mismatches that require thoughtful outreach:
- Almost-qualified (missing one key requirement)
- Overqualified (director applying for IC role)
- Career changers (relevant transferable skills but different domain)
- Location/visa mismatches

Usage:
  python data/augment_candidates.py --roles data/roles.json --count 50
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CANDIDATES = BASE_DIR / "data" / "candidates.json"
DEFAULT_ROLES = BASE_DIR / "data" / "roles.json"
DEFAULT_FORMATTED = BASE_DIR / "data" / "candidates_formatted.jsonl"


# =============================================================================
# ROLE CLASSIFICATION - Map roles to appropriate candidate types
# =============================================================================

ROLE_CATEGORIES = {
    "finance_tutor": [
        "finance tutor", "economics tutor", "accounting", "banking", 
        "portfolio", "quantitative finance", "sell-side", "personal finance"
    ],
    "stem_tutor": [
        "biology tutor", "chemistry tutor", "physics tutor", "math tutor",
        "statistics tutor", "pure math", "applied math", "space science",
        "earth science", "materials science", "medicine tutor"
    ],
    "engineering_tutor": [
        "mechanical engineering tutor", "electrical engineering tutor",
        "civil engineering tutor", "chemical engineering tutor",
        "systems engineering tutor", "data science tutor"
    ],
    "creative_tutor": [
        "audio tutor", "image tutor", "video tutor", "video games tutor",
        "web design tutor", "presentation", "writing tutor", "memes",
        "personality", "multilingual tutor"
    ],
    "legal_tutor": ["legal", "compliance tutor"],
    "healthcare_tutor": ["healthcare tutor", "administration tutor"],
    "ml_engineer": [
        "member of technical staff", "mts", "pre-training", "post-training",
        "inference", "multimodal", "reasoning", "rl ", "cuda", "jax",
        "image generation", "video generation", "world model"
    ],
    "backend_engineer": ["backend engineer", "rust", "c++"],
    "frontend_engineer": ["frontend engineer", "react", "design engineer"],
    "fullstack_engineer": ["fullstack", "product engineer"],
    "infra_engineer": [
        "infrastructure", "sre", "site reliability", "platform",
        "supercomputing", "storage", "observability", "reliability"
    ],
    "network_engineer": ["network", "rdma", "networking", "hpc network"],
    "security_engineer": [
        "security engineer", "appsec", "detection", "infrasec",
        "cybersecurity"
    ],
    "data_engineer": ["data platform", "data acquisition", "crawling"],
    "mobile_engineer": ["mobile", "ios", "android"],
    "operations": [
        "datacenter", "facilities", "maintenance", "technician",
        "construction", "fiber", "operations"
    ],
    "business": [
        "recruiter", "hr ", "people operations", "accountant", "controller",
        "fp&a", "revenue", "sales", "client partner", "growth"
    ],
    "specialist": [
        "specialist", "barista", "security specialist", "ambassador"
    ],
}


def classify_role(title: str) -> str:
    """Classify a role title into a candidate category."""
    t = title.lower()
    for category, keywords in ROLE_CATEGORIES.items():
        if any(k in t for k in keywords):
            return category
    # Default fallback
    if "tutor" in t:
        return "stem_tutor"
    if "engineer" in t:
        return "backend_engineer"
    return "ml_engineer"


# =============================================================================
# NAME GENERATION
# =============================================================================

FIRST_NAMES = [
    "Ava", "Noah", "Liam", "Mia", "Ethan", "Sofia", "Leo", "Isla", "Maya", "Jude",
    "Iris", "Caleb", "Aria", "Nico", "Zara", "Kai", "Elena", "Mason", "Nova", "Elias",
    "Priya", "Ravi", "Sanaa", "Mateo", "Yara", "Samir", "Anya", "Dante", "Lucia", "Rowan",
    "Felix", "Camila", "Omar", "Selene", "Ada", "Hugo", "Imani", "Jonas", "Tara", "Zane",
    "Wei", "Aisha", "Dmitri", "Kenji", "Fatima", "Olga", "Henrik", "Yuki", "Rashid", "Linnea",
    "Margaret", "William", "James", "Sarah", "Michael", "Jennifer", "Robert", "Linda",
    "David", "Elizabeth", "Richard", "Barbara", "Charles", "Susan", "Joseph", "Jessica",
]

LAST_NAMES = [
    "Chen", "Patel", "Garcia", "Nguyen", "Brown", "Davis", "Khan", "Yamamoto", "Kim",
    "Singh", "Lopez", "Martinez", "Silva", "Rossi", "Ivanov", "Williams", "Jones",
    "Rahman", "Hernandez", "Smith", "Kaur", "Ali", "Sato", "Dubois", "Kowalski",
    "Costa", "Meier", "Andersson", "Hughes", "Murphy", "Zhang", "Okonkwo", "Petrov",
    "Thompson", "White", "Harris", "Clark", "Lewis", "Robinson", "Walker", "Young",
]


def synth_name() -> str:
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


# =============================================================================
# LOCATION HANDLING
# =============================================================================

LOCATIONS_US_TECH = ["San Francisco, CA", "Palo Alto, CA", "Seattle, WA", "New York, NY", "Austin, TX"]
LOCATIONS_US_OTHER = ["Memphis, TN", "Chicago, IL", "Boston, MA", "Denver, CO", "Phoenix, AZ"]
LOCATIONS_INTERNATIONAL = [
    "London, UK", "Berlin, Germany", "Tokyo, Japan", "Singapore", "Toronto, Canada",
    "Sydney, Australia", "Amsterdam, Netherlands", "Dublin, Ireland", "Tel Aviv, Israel",
    "Bangalore, India", "São Paulo, Brazil", "Stockholm, Sweden", "Paris, France",
]
LOCATIONS_REMOTE = ["Remote (US)", "Remote (EU timezone)", "Remote (flexible)"]


def pick_location(role: Dict[str, Any], mismatch: bool = False) -> Tuple[str, str, bool]:
    """
    Pick candidate location. Returns (location, timezone, has_visa_issue).
    mismatch=True creates intentional location/visa problems.
    """
    role_loc = (role.get("location") or "").lower()
    
    if mismatch:
        # Create visa/location issues
        if random.random() < 0.5:
            loc = random.choice(LOCATIONS_INTERNATIONAL)
            return loc, "Not US timezone", True
        else:
            # Wrong US location for onsite role
            if "remote" not in role_loc:
                return random.choice(LOCATIONS_INTERNATIONAL), "EU/APAC timezone", True
    
    # Match the role location
    if "remote" in role_loc:
        return random.choice(LOCATIONS_REMOTE + LOCATIONS_US_TECH), "Flexible", False
    if "memphis" in role_loc:
        return random.choice(["Memphis, TN", "Nashville, TN", "Remote (willing to relocate)"]), "CST", False
    if "london" in role_loc or "dublin" in role_loc:
        return random.choice(["London, UK", "Dublin, Ireland", "Remote (UK)"]), "GMT/BST", False
    
    return random.choice(LOCATIONS_US_TECH), "PST", False


# =============================================================================
# DOMAIN-SPECIFIC PROFILE GENERATORS
# =============================================================================

# ----- FINANCE/ECONOMICS TUTORS -----

FINANCE_SPECIALIZATIONS = [
    "macroeconomic policy", "monetary economics", "behavioral finance",
    "derivatives pricing", "portfolio theory", "corporate finance",
    "financial accounting", "investment banking", "equity research",
    "quantitative trading", "risk management", "fixed income",
]

FINANCE_CREDENTIALS = [
    "PhD Economics, MIT", "PhD Finance, Wharton", "PhD Economics, Chicago",
    "PhD Economics, Stanford", "PhD Economics, Harvard", "PhD Finance, NYU Stern",
    "CFA Charterholder + MBA, Columbia", "PhD Financial Economics, LSE",
]

FINANCE_INSTITUTIONS = [
    "Federal Reserve Board", "IMF", "World Bank", "Goldman Sachs Research",
    "Morgan Stanley", "JPMorgan", "BlackRock", "Bridgewater Associates",
    "Citadel", "Two Sigma", "AQR Capital", "PIMCO",
]

FINANCE_PUBLICATIONS = [
    "Journal of Finance", "Quarterly Journal of Economics", "Review of Financial Studies",
    "American Economic Review", "Journal of Monetary Economics", "Journal of Financial Economics",
]


def build_finance_tutor_profile() -> Dict[str, Any]:
    spec = random.choice(FINANCE_SPECIALIZATIONS)
    institution = random.choice(FINANCE_INSTITUTIONS)
    years = random.randint(5, 20)
    
    return {
        "domain": "finance",
        "education": random.choice(FINANCE_CREDENTIALS),
        "current_title": random.choice([
            f"Senior Economist at {institution}",
            f"Associate Professor of Economics",
            f"Director of Research at {institution}",
            f"Portfolio Manager at {institution}",
            f"Quantitative Researcher at {institution}",
        ]),
        "experience_years": years,
        "specialization": spec,
        "publications": [
            f"'{random.choice(['The Impact of', 'Modeling', 'A New Approach to', 'Empirical Analysis of'])} {spec.title()}' - {random.choice(FINANCE_PUBLICATIONS)} ({random.randint(2015, 2024)})",
        ] if random.random() > 0.3 else [],
        "skills": random.sample([
            "Econometrics", "Time Series Analysis", "Python", "R", "Stata",
            "Bloomberg Terminal", "Financial Modeling", "Valuation",
            "Monte Carlo Simulation", "Factor Models", "GARCH Models",
        ], k=random.randint(4, 6)),
        "highlights": [
            f"{years} years in {spec}",
            f"Previously at {random.choice(FINANCE_INSTITUTIONS)}",
            random.choice([
                "Testified before Congress on monetary policy",
                "Managed $2B+ AUM portfolio",
                "Published 15+ peer-reviewed papers",
                "Developed proprietary trading models",
                "Taught graduate-level finance courses",
            ]),
        ],
    }


# ----- STEM TUTORS (Biology, Chemistry, Physics, Math) -----

STEM_FIELDS = {
    "biology": {
        "specs": ["molecular biology", "genetics", "immunology", "neuroscience", "cell biology", "bioinformatics"],
        "journals": ["Nature", "Cell", "Science", "PNAS", "Nature Genetics"],
        "institutions": ["NIH", "Broad Institute", "Cold Spring Harbor", "Salk Institute", "HHMI"],
        "degrees": ["PhD Biology, Harvard", "PhD Molecular Biology, MIT", "PhD Genetics, Stanford", "MD-PhD, UCSF"],
    },
    "chemistry": {
        "specs": ["organic chemistry", "biochemistry", "physical chemistry", "computational chemistry", "materials chemistry"],
        "journals": ["JACS", "Nature Chemistry", "Angewandte Chemie", "Chemical Reviews"],
        "institutions": ["Dow Chemical", "BASF", "Merck Research", "Pfizer", "Genentech"],
        "degrees": ["PhD Chemistry, Caltech", "PhD Organic Chemistry, MIT", "PhD Chemical Biology, Harvard"],
    },
    "physics": {
        "specs": ["quantum mechanics", "condensed matter", "particle physics", "astrophysics", "computational physics"],
        "journals": ["Physical Review Letters", "Nature Physics", "Science", "JHEP"],
        "institutions": ["CERN", "Fermilab", "SLAC", "Los Alamos", "NASA JPL", "Bell Labs"],
        "degrees": ["PhD Physics, Princeton", "PhD Physics, Caltech", "PhD Theoretical Physics, Cambridge"],
    },
    "math": {
        "specs": ["number theory", "topology", "differential geometry", "probability theory", "combinatorics", "analysis"],
        "journals": ["Annals of Mathematics", "Inventiones", "JAMS", "Acta Mathematica"],
        "institutions": ["IAS Princeton", "MSRI", "Clay Mathematics", "Fields Institute"],
        "degrees": ["PhD Mathematics, Princeton", "PhD Mathematics, MIT", "PhD Mathematics, Berkeley"],
    },
    "medicine": {
        "specs": ["internal medicine", "cardiology", "oncology", "neurology", "emergency medicine", "surgery"],
        "journals": ["NEJM", "Lancet", "JAMA", "BMJ", "Annals of Internal Medicine"],
        "institutions": ["Mayo Clinic", "Johns Hopkins", "Cleveland Clinic", "Mass General", "UCSF Medical"],
        "degrees": ["MD, Harvard Medical School", "MD-PhD, Johns Hopkins", "MD, Stanford", "MD, UCSF"],
    },
}


def build_stem_tutor_profile(field: str = "biology") -> Dict[str, Any]:
    field_data = STEM_FIELDS.get(field, STEM_FIELDS["biology"])
    spec = random.choice(field_data["specs"])
    years = random.randint(8, 25)
    
    return {
        "domain": field,
        "education": random.choice(field_data["degrees"]),
        "current_title": random.choice([
            f"Professor of {field.title()} at {random.choice(['Stanford', 'MIT', 'Harvard', 'Berkeley', 'Caltech'])}",
            f"Research Scientist at {random.choice(field_data['institutions'])}",
            f"Senior Scientist at {random.choice(field_data['institutions'])}",
            f"Associate Professor, {field.title()} Department",
        ]),
        "experience_years": years,
        "specialization": spec,
        "publications": [
            f"'{random.choice(['Novel', 'Characterization of', 'Mechanisms of', 'Discovery of'])} {spec.title()}' - {random.choice(field_data['journals'])} ({random.randint(2018, 2024)})",
        ],
        "skills": random.sample([
            "Research Design", "Grant Writing", "Statistical Analysis", "Python", "R",
            "Lab Management", "Scientific Writing", "Peer Review", "Teaching",
        ], k=random.randint(4, 6)),
        "highlights": [
            f"{years} years research experience in {spec}",
            f"Published in {random.choice(field_data['journals'])}",
            random.choice([
                "H-index of 35+",
                "NIH R01 grant recipient",
                "Graduate thesis advisor for 10+ PhDs",
                "Department teaching award recipient",
                "Keynote speaker at international conferences",
            ]),
        ],
    }


# ----- ML/AI ENGINEERS -----

ML_COMPANIES = [
    "OpenAI", "Anthropic", "DeepMind", "Google Brain", "Meta AI", "Cohere",
    "Scale AI", "Hugging Face", "Weights & Biases", "Databricks", "Nvidia",
]

ML_SPECIALIZATIONS = [
    "large language models", "reinforcement learning", "multimodal models",
    "inference optimization", "distributed training", "model alignment",
    "computer vision", "speech recognition", "recommendation systems",
]


def build_ml_engineer_profile() -> Dict[str, Any]:
    spec = random.choice(ML_SPECIALIZATIONS)
    company = random.choice(ML_COMPANIES)
    years = random.randint(3, 12)
    
    return {
        "domain": "ml_engineering",
        "education": random.choice([
            "PhD Machine Learning, CMU", "MS Computer Science, Stanford",
            "PhD AI, Berkeley", "MS ML, MIT", "BS CS, Stanford (dropped out of PhD)",
        ]),
        "current_title": random.choice([
            f"Senior ML Engineer at {company}",
            f"Research Engineer at {company}",
            f"Staff ML Engineer at {company}",
            f"ML Platform Lead at {company}",
        ]),
        "experience_years": years,
        "specialization": spec,
        "skills": random.sample([
            "PyTorch", "JAX", "CUDA", "Triton", "vLLM", "TensorRT",
            "Distributed Training", "RLHF", "LoRA", "Ray", "Kubernetes",
        ], k=random.randint(5, 8)),
        "highlights": [
            f"Led {spec} team at {company}",
            random.choice([
                "Reduced inference latency by 60%",
                "Scaled training to 10K GPUs",
                "Shipped model serving 100M+ users",
                "Published at NeurIPS/ICML",
                "Open-sourced popular ML library (5K+ stars)",
            ]),
            f"{years} years in production ML systems",
        ],
    }


# ----- BACKEND/INFRA ENGINEERS -----

TECH_COMPANIES = [
    "Google", "Meta", "Amazon", "Microsoft", "Netflix", "Uber", "Stripe",
    "Airbnb", "Dropbox", "Cloudflare", "Datadog", "Snowflake", "Confluent",
]


def build_backend_engineer_profile() -> Dict[str, Any]:
    company = random.choice(TECH_COMPANIES)
    years = random.randint(4, 15)
    
    return {
        "domain": "backend_engineering",
        "education": random.choice([
            "BS Computer Science, Stanford", "MS CS, MIT", "BS CS, Berkeley",
            "MS Distributed Systems, CMU", "BS CS, Georgia Tech",
        ]),
        "current_title": random.choice([
            f"Senior Software Engineer at {company}",
            f"Staff Engineer at {company}",
            f"Backend Tech Lead at {company}",
            f"Principal Engineer at {company}",
        ]),
        "experience_years": years,
        "specialization": random.choice([
            "distributed systems", "API design", "database systems",
            "microservices", "high-performance computing",
        ]),
        "skills": random.sample([
            "Go", "Rust", "Python", "C++", "Kubernetes", "PostgreSQL",
            "Redis", "Kafka", "gRPC", "GraphQL", "AWS", "GCP",
        ], k=random.randint(5, 8)),
        "highlights": [
            f"Built systems handling {random.choice(['1M', '10M', '100M'])} QPS",
            f"{years} years at top tech companies",
            random.choice([
                "Designed company-wide API gateway",
                "Led migration from monolith to microservices",
                "Reduced p99 latency from 500ms to 50ms",
                "Built real-time data pipeline processing 1TB/day",
            ]),
        ],
    }


def build_infra_engineer_profile() -> Dict[str, Any]:
    company = random.choice(TECH_COMPANIES + ["AWS", "GCP", "Azure"])
    years = random.randint(5, 15)
    
    return {
        "domain": "infrastructure",
        "education": random.choice([
            "BS Computer Science, MIT", "MS Systems, Stanford",
            "BS CS, Berkeley", "MS CS, CMU",
        ]),
        "current_title": random.choice([
            f"Senior SRE at {company}",
            f"Staff Infrastructure Engineer at {company}",
            f"Platform Tech Lead at {company}",
            f"Principal SRE at {company}",
        ]),
        "experience_years": years,
        "specialization": random.choice([
            "Kubernetes", "storage systems", "observability",
            "reliability engineering", "cloud infrastructure",
        ]),
        "skills": random.sample([
            "Kubernetes", "Terraform", "Prometheus", "Linux", "Python",
            "Go", "AWS", "GCP", "Ansible", "Docker", "Helm",
        ], k=random.randint(5, 8)),
        "highlights": [
            f"Maintained {random.choice(['99.99%', '99.999%'])} uptime for critical services",
            f"{years} years in infrastructure/SRE",
            random.choice([
                "Built Kubernetes platform serving 500+ engineers",
                "Led incident response for $1M+/hour services",
                "Reduced cloud costs by 40%",
                "Designed multi-region disaster recovery",
            ]),
        ],
    }


# ----- OPERATIONS/FACILITIES -----

def build_operations_profile() -> Dict[str, Any]:
    years = random.randint(3, 20)
    spec = random.choice([
        "datacenter operations", "facilities management", "construction",
        "electrical systems", "HVAC systems", "fiber installation",
    ])
    
    return {
        "domain": "operations",
        "education": random.choice([
            "AS Electrical Technology", "BS Facilities Management",
            "Journeyman Electrician License", "BS Construction Management",
            "Technical Certificate, HVAC", "High school + 10 years experience",
        ]),
        "current_title": random.choice([
            f"Senior Datacenter Technician at {random.choice(['Equinix', 'Digital Realty', 'CoreSite'])}",
            f"Facilities Manager at {random.choice(['Google', 'Meta', 'Amazon'])}",
            f"Construction Supervisor at {random.choice(['Turner', 'Skanska', 'DPR'])}",
            f"Operations Lead at {random.choice(['AWS', 'Microsoft', 'Oracle'])} datacenter",
        ]),
        "experience_years": years,
        "specialization": spec,
        "skills": random.sample([
            "Electrical Systems", "HVAC", "Fire Suppression", "BMS",
            "Safety Protocols", "Vendor Management", "Preventive Maintenance",
            "OSHA Certified", "Forklift Certified", "CPR/First Aid",
        ], k=random.randint(4, 6)),
        "highlights": [
            f"{years} years in {spec}",
            random.choice([
                "Zero safety incidents in 5+ years",
                "Managed team of 15+ technicians",
                "Completed $50M facility buildout",
                "Reduced downtime by 70%",
            ]),
        ],
    }


# ----- BUSINESS/HR -----

def build_business_profile() -> Dict[str, Any]:
    years = random.randint(4, 15)
    spec = random.choice([
        "HR business partnering", "technical recruiting", "people operations",
        "accounting", "FP&A", "revenue operations",
    ])
    
    companies = ["Google", "Meta", "Netflix", "Stripe", "Airbnb", "Salesforce"]
    
    return {
        "domain": "business",
        "education": random.choice([
            "MBA, Wharton", "BS Business Administration, Berkeley",
            "MS HRM, Cornell", "CPA + BS Accounting, NYU",
            "MBA, Kellogg", "BA Psychology, Stanford",
        ]),
        "current_title": random.choice([
            f"Senior HRBP at {random.choice(companies)}",
            f"Technical Recruiter at {random.choice(companies)}",
            f"Senior Accountant at {random.choice(companies)}",
            f"FP&A Manager at {random.choice(companies)}",
            f"People Operations Lead at {random.choice(companies)}",
        ]),
        "experience_years": years,
        "specialization": spec,
        "skills": random.sample([
            "Workday", "Greenhouse", "Excel", "SQL", "People Analytics",
            "Compensation Design", "GAAP", "NetSuite", "Tableau",
        ], k=random.randint(4, 6)),
        "highlights": [
            f"{years} years in {spec}",
            random.choice([
                "Scaled org from 50 to 500 employees",
                "Hired 100+ engineers in 12 months",
                "Led M&A integration for 3 acquisitions",
                "Implemented new HRIS for 2000+ employees",
                "Closed books for IPO",
            ]),
        ],
    }


# ----- CREATIVE/CONTENT TUTORS -----

def build_creative_tutor_profile() -> Dict[str, Any]:
    spec = random.choice([
        "audio production", "video editing", "game design",
        "web design", "UX writing", "content strategy",
    ])
    years = random.randint(5, 20)
    
    return {
        "domain": "creative",
        "education": random.choice([
            "MFA Film Production, USC", "BFA Graphic Design, RISD",
            "BA Music Production, Berklee", "MS HCI, Carnegie Mellon",
            "BFA Game Design, NYU", "Self-taught, 15 years professional experience",
        ]),
        "current_title": random.choice([
            f"Senior {spec.title().replace('_', ' ')} at {random.choice(['Spotify', 'Netflix', 'YouTube', 'Adobe'])}",
            f"Creative Director at {random.choice(['IDEO', 'Pentagram', 'frog design'])}",
            f"Lead Designer at {random.choice(['Figma', 'Canva', 'InVision'])}",
        ]),
        "experience_years": years,
        "specialization": spec,
        "skills": random.sample([
            "Adobe Creative Suite", "Figma", "Pro Tools", "Final Cut",
            "Unity", "Unreal Engine", "Blender", "After Effects",
        ], k=random.randint(4, 6)),
        "highlights": [
            f"{years} years in {spec}",
            random.choice([
                "Emmy-nominated for documentary work",
                "Designed products used by 50M+ users",
                "Grammy-winning album credits",
                "Shipped 3 AAA game titles",
                "Built design system adopted company-wide",
            ]),
        ],
    }


# ----- LEGAL TUTOR -----

def build_legal_tutor_profile() -> Dict[str, Any]:
    spec = random.choice([
        "corporate law", "IP law", "regulatory compliance",
        "privacy law", "employment law", "securities law",
    ])
    years = random.randint(8, 25)
    
    return {
        "domain": "legal",
        "education": random.choice([
            "JD, Harvard Law School", "JD, Yale Law School", "JD, Stanford Law",
            "JD, Columbia Law + LLM Tax", "JD, NYU Law",
        ]),
        "current_title": random.choice([
            f"Partner at {random.choice(['Skadden', 'Sullivan & Cromwell', 'Cravath', 'Wachtell'])}",
            f"General Counsel at {random.choice(['Stripe', 'Airbnb', 'DoorDash'])}",
            f"VP Legal at {random.choice(['Google', 'Meta', 'Microsoft'])}",
            f"Of Counsel at {random.choice(['Wilson Sonsini', 'Cooley', 'Fenwick'])}",
        ]),
        "experience_years": years,
        "specialization": spec,
        "bar_admissions": random.sample(["California", "New York", "Delaware", "DC"], k=random.randint(1, 3)),
        "skills": ["Legal Research", "Contract Drafting", "Regulatory Compliance", "M&A", "Litigation"],
        "highlights": [
            f"{years} years practicing {spec}",
            random.choice([
                "Led $10B+ M&A transactions",
                "Argued before federal appellate courts",
                "Built compliance program for public company",
                "Managed team of 20+ attorneys",
            ]),
        ],
    }


# =============================================================================
# EDGE CASE GENERATORS - Create intentional mismatches
# =============================================================================

EDGE_CASE_TYPES = [
    "overqualified",          # Director applying for IC role
    "underqualified",         # Missing key requirement (e.g., no PhD for tutor role)
    "career_changer",         # Different domain but transferable skills
    "location_mismatch",      # Wrong location/timezone
    "seniority_gap",          # Wrong level for the role
    # --- NEW: Harder edge cases ---
    "domain_mismatch_subtle", # Impressive credentials in adjacent-but-wrong domain
    "sparse_profile",         # Minimal info - forces LLM to work with less
    "red_flags",              # Job hopping, buzzword-heavy, unexplained gaps
    "impressive_but_irrelevant",  # Great accomplishments that don't transfer
    "outdated_experience",    # Relevant experience but 5+ years stale
    "overly_academic",        # PhD researcher with no industry/applied experience
    "wrong_specialization",   # Right field, wrong sub-specialty for the role
]


def apply_edge_case(profile: Dict[str, Any], edge_type: str, role: Dict[str, Any]) -> Dict[str, Any]:
    """Modify a profile to create a specific type of mismatch."""
    
    if edge_type == "overqualified":
        profile["seniority"] = random.choice(["VP", "Director", "C-level", "Partner"])
        profile["edge_case_note"] = "Overqualified - senior executive applying for IC/senior IC role"
        profile["highlights"].append(random.choice([
            "Currently VP-level, seeking IC role for better work-life balance",
            "Former CEO of startup, looking to return to hands-on work",
            "Director-level, interested in moving to high-growth company",
        ]))
    
    elif edge_type == "underqualified":
        # Downgrade education/experience
        if profile.get("domain") in ["finance", "biology", "chemistry", "physics", "math", "medicine", "legal"]:
            profile["education"] = random.choice([
                "BS in related field (no graduate degree)",
                "Self-taught, no formal education",
                "Currently enrolled in PhD program (not complete)",
                "MS only (role requires PhD)",
            ])
            profile["edge_case_note"] = "Missing key qualification (e.g., PhD required but not held)"
        else:
            profile["experience_years"] = random.randint(1, 2)
            profile["edge_case_note"] = "Junior candidate for senior role"
    
    elif edge_type == "career_changer":
        original_domain = profile.get("domain", "unknown")
        new_summary = random.choice([
            f"Transitioning from {original_domain} to AI/ML after completing bootcamp",
            f"Former {original_domain} professional pivoting to tech",
            f"Career changer with 10 years in {original_domain}, learning new domain",
        ])
        profile["edge_case_note"] = f"Career changer from {original_domain}"
        profile["highlights"].append(new_summary)
    
    elif edge_type == "location_mismatch":
        role_loc = role.get("location", "")
        if "remote" not in role_loc.lower():
            profile["location"] = random.choice(LOCATIONS_INTERNATIONAL)
            profile["timezone"] = "Not US timezone"
            profile["has_visa_issue"] = True
            profile["edge_case_note"] = "Location/visa mismatch - international candidate for US onsite role"
    
    elif edge_type == "seniority_gap":
        role_title = role.get("title", "").lower()
        if "senior" in role_title or "staff" in role_title or "principal" in role_title:
            profile["experience_years"] = random.randint(1, 3)
            profile["current_title"] = profile["current_title"].replace("Senior ", "").replace("Staff ", "").replace("Principal ", "")
            profile["edge_case_note"] = "Junior candidate for senior role"
        else:
            profile["experience_years"] = random.randint(15, 25)
            profile["seniority"] = "Director/VP level"
            profile["edge_case_note"] = "Very senior candidate for entry/mid role"
    
    # --- NEW HARDER EDGE CASES ---
    
    elif edge_type == "domain_mismatch_subtle":
        # Impressive credentials but in adjacent-but-wrong domain
        # e.g., PhD computational biology for a pure math tutor role
        original_domain = profile.get("domain", "unknown")
        adjacent_domains = {
            "finance": ["economics research (not trading)", "data science (no finance)", "statistics (academic)"],
            "biology": ["bioinformatics (no wet lab)", "computational modeling (no biology)", "chemistry (no bio)"],
            "chemistry": ["materials science", "chemical engineering (process, not synthesis)", "physics"],
            "physics": ["applied math", "engineering physics", "astronomy (observational)"],
            "math": ["computer science theory", "statistics (applied)", "quantitative social science"],
            "ml_engineering": ["academic ML research (no production)", "data science (no ML ops)", "software (no ML)"],
            "backend_engineering": ["data engineering (no backend)", "devops (no coding)", "QA automation"],
            "infrastructure": ["software engineering (no infra)", "security (no SRE)", "networking (no cloud)"],
        }
        adjacent = adjacent_domains.get(original_domain, ["adjacent field"])
        profile["specialization"] = random.choice(adjacent)
        profile["edge_case_note"] = f"Subtle domain mismatch: {original_domain} background but specialization is {profile['specialization']}"
        # Remove some relevant skills
        if profile.get("skills"):
            profile["skills"] = profile["skills"][:2] + ["Adjacent skill not directly applicable"]
    
    elif edge_type == "sparse_profile":
        # Minimal information - forces LLM to work with less
        profile["highlights"] = [random.choice(profile.get("highlights", ["Experience in field"]))]
        profile["publications"] = []
        profile["skills"] = random.sample(profile.get("skills", []), k=min(2, len(profile.get("skills", []))))
        profile["specialization"] = ""  # Remove specialization
        profile["education"] = profile.get("education", "").split(",")[0]  # Just degree, no school
        profile["edge_case_note"] = "Sparse profile - minimal information to work with"
    
    elif edge_type == "red_flags":
        # Job hopping, buzzword-heavy, unexplained gaps
        red_flag_type = random.choice(["job_hopper", "buzzword_heavy", "gaps", "vague_accomplishments"])
        
        if red_flag_type == "job_hopper":
            profile["experience_years"] = random.randint(6, 10)
            profile["highlights"] = [
                f"5 roles in {profile['experience_years']} years",
                "Most recent tenure: 8 months",
                random.choice(profile.get("highlights", ["Worked on various projects"])),
            ]
            profile["edge_case_note"] = "Red flag: Job hopper - 5 roles in short period"
        
        elif red_flag_type == "buzzword_heavy":
            profile["highlights"] = [
                "Leveraged synergies to drive innovation and stakeholder alignment",
                "Passionate about disrupting paradigms with cutting-edge solutions",
                "Strategic thought leader with proven track record of excellence",
            ]
            profile["skills"] = ["Thought Leadership", "Strategic Vision", "Change Management", "Stakeholder Engagement"]
            profile["edge_case_note"] = "Red flag: Buzzword-heavy profile with no concrete accomplishments"
        
        elif red_flag_type == "gaps":
            profile["highlights"] = [
                f"Last role ended {random.randint(2, 4)} years ago",
                "Currently taking time off",
                random.choice(profile.get("highlights", ["Prior experience in field"])),
            ]
            profile["edge_case_note"] = "Red flag: Significant employment gap"
        
        else:  # vague_accomplishments
            profile["highlights"] = [
                "Worked on important projects",
                "Contributed to team success",
                "Helped improve processes",
            ]
            profile["edge_case_note"] = "Red flag: Vague accomplishments with no metrics or specifics"
    
    elif edge_type == "impressive_but_irrelevant":
        # Great accomplishments that don't transfer
        irrelevant_highlights = {
            "finance": [
                "Olympic gold medalist in swimming",
                "Published novelist with 3 NYT bestsellers",
                "Founded successful restaurant chain (10 locations)",
            ],
            "ml_engineering": [
                "Professional poker player (WSOP finalist)",
                "Real estate investor ($50M portfolio)",
                "Former professional athlete (5 years NFL)",
            ],
            "default": [
                "YouTube channel with 2M subscribers (cooking content)",
                "Built and sold e-commerce business for $5M",
                "Competitive chess grandmaster",
            ],
        }
        domain = profile.get("domain", "default")
        irrelevant = irrelevant_highlights.get(domain, irrelevant_highlights["default"])
        # Keep one relevant highlight, add irrelevant ones
        original_highlight = profile.get("highlights", ["Experience"])[0] if profile.get("highlights") else "Some experience"
        profile["highlights"] = [original_highlight] + random.sample(irrelevant, k=2)
        profile["edge_case_note"] = "Impressive but irrelevant accomplishments - great achievements that don't transfer"
    
    elif edge_type == "outdated_experience":
        # Relevant experience but 5+ years stale
        years_ago = random.randint(5, 12)
        profile["highlights"] = [
            f"Left {profile.get('domain', 'the field')} {years_ago} years ago",
            f"Recent work: {random.choice(['consulting', 'teaching', 'startup founder (different domain)', 'sabbatical'])}",
            f"Core experience is from {2024 - years_ago - profile.get('experience_years', 5)}-{2024 - years_ago}",
        ]
        profile["current_title"] = random.choice([
            "Independent Consultant",
            "Startup Founder (stealth)",
            "Career Break",
            f"Adjunct Professor (teaching {profile.get('domain', 'various topics')})",
        ])
        profile["edge_case_note"] = f"Outdated experience - relevant work was {years_ago}+ years ago"
    
    elif edge_type == "overly_academic":
        # PhD researcher with no industry/applied experience
        profile["experience_years"] = random.randint(8, 15)
        profile["current_title"] = f"Associate Professor at {random.choice(['State University', 'Regional College', 'Community College'])}"
        profile["highlights"] = [
            f"{profile['experience_years']} years in academia only - no industry experience",
            "Published 40+ papers (citation count: 150)",
            "Never worked outside university setting",
            "Teaching load: 4 courses per semester",
        ]
        profile["skills"] = [s for s in profile.get("skills", []) if s not in ["Python", "Production Systems", "Kubernetes"]]
        profile["skills"].extend(["Academic Writing", "Grant Applications", "Committee Service"])
        profile["edge_case_note"] = "Overly academic - extensive research but zero industry/applied experience"
    
    elif edge_type == "wrong_specialization":
        # Right field, wrong sub-specialty
        wrong_specs = {
            "finance": {
                "for_quant_role": ["retail banking", "compliance", "wealth management (HNW individuals)"],
                "for_tutor_role": ["sales & trading (no research)", "investment banking (M&A, not analysis)", "private banking"],
            },
            "ml_engineering": {
                "for_llm_role": ["computer vision only", "robotics/controls", "speech (not text)"],
                "for_infra_role": ["research only (no systems)", "NLP theory (no production)", "RL for games"],
            },
            "biology": {
                "for_molecular_role": ["ecology/field biology", "marine biology", "evolutionary biology (no lab)"],
                "for_tutor_role": ["agricultural science", "forensic biology", "wildlife conservation"],
            },
        }
        domain = profile.get("domain", "")
        if domain in wrong_specs:
            spec_category = random.choice(list(wrong_specs[domain].keys()))
            profile["specialization"] = random.choice(wrong_specs[domain][spec_category])
            profile["edge_case_note"] = f"Wrong specialization: {domain} background but specializes in {profile['specialization']} (not what role needs)"
        else:
            profile["specialization"] = "tangentially related subspecialty"
            profile["edge_case_note"] = "Wrong specialization within the field"
    
    return profile


# =============================================================================
# MAIN CANDIDATE BUILDER
# =============================================================================

def build_profile_for_role_category(category: str) -> Dict[str, Any]:
    """Generate an appropriate profile for a role category."""
    
    if category == "finance_tutor":
        return build_finance_tutor_profile()
    elif category == "stem_tutor":
        field = random.choice(["biology", "chemistry", "physics", "math", "medicine"])
        return build_stem_tutor_profile(field)
    elif category == "engineering_tutor":
        return build_stem_tutor_profile("math")  # Engineering tutors often have math/physics backgrounds
    elif category == "creative_tutor":
        return build_creative_tutor_profile()
    elif category == "legal_tutor":
        return build_legal_tutor_profile()
    elif category == "healthcare_tutor":
        return build_stem_tutor_profile("medicine")
    elif category == "ml_engineer":
        return build_ml_engineer_profile()
    elif category in ["backend_engineer", "frontend_engineer", "fullstack_engineer", 
                      "mobile_engineer", "data_engineer", "security_engineer"]:
        return build_backend_engineer_profile()
    elif category in ["infra_engineer", "network_engineer"]:
        return build_infra_engineer_profile()
    elif category == "operations":
        return build_operations_profile()
    elif category in ["business", "specialist"]:
        return build_business_profile()
    else:
        return build_backend_engineer_profile()


def build_candidate(role: Dict[str, Any], cid: int, edge_case_pct: float = 0.55) -> Dict[str, Any]:
    """Build a candidate with appropriate (or intentionally mismatched) background.
    
    Default edge_case_pct increased to 0.55 (was 0.30) to create more challenging
    eval scenarios. Real recruiting involves many imperfect-fit candidates.
    """
    
    role_title = role.get("title", "")
    category = classify_role(role_title)
    
    # Determine if this should be an edge case
    is_edge_case = random.random() < edge_case_pct
    edge_type = None
    
    if is_edge_case:
        # Weight newer/harder edge cases more heavily
        weighted_types = (
            EDGE_CASE_TYPES[:5]  # Original types
            + EDGE_CASE_TYPES[5:] * 2  # New harder types appear 2x as often
        )
        edge_type = random.choice(weighted_types)
    
    # Generate base profile appropriate for the role
    profile = build_profile_for_role_category(category)
    
    # Get location
    loc, tz, visa_issue = pick_location(role, mismatch=(edge_type == "location_mismatch"))
    
    # Apply edge case modifications
    if is_edge_case and edge_type:
        profile = apply_edge_case(profile, edge_type, role)
    
    name = synth_name()
    
    # Determine job search status - mix of active/passive
    openness = random.choice([
        "Actively looking",
        "Open to opportunities",
        "Not actively looking but always curious",
        "Happy at current role, open to exceptional opportunities",
        "Exploring after recent layoff",
        "Passively interested in AI/ML companies",
        "Looking to relocate",
        "Open to remote or hybrid",
    ])
    
    # Build final candidate object
    candidate = {
        "id": f"cand-{cid:03d}",
        "name": name,
        "target_role_title": role_title,
        "target_role_url": role.get("absolute_url"),
        "location": profile.get("location", loc),
        "timezone": profile.get("timezone", tz),
        "has_visa_issue": profile.get("has_visa_issue", visa_issue),
        
        # Profile details
        "current_title": profile.get("current_title", ""),
        "education": profile.get("education", ""),
        "experience_years": profile.get("experience_years", 5),
        "domain": profile.get("domain", category),
        "specialization": profile.get("specialization", ""),
        
        "skills": profile.get("skills", []),
        "highlights": profile.get("highlights", []),
        "publications": profile.get("publications", []),
        
        "openness": openness,
        
        # Edge case tracking
        "is_edge_case": is_edge_case,
        "edge_case_type": edge_type,
        "edge_case_note": profile.get("edge_case_note", ""),
        
        # Include JD excerpt for context
        "job_description_excerpt": (role.get("content_text") or role.get("content_html") or "")[:1500].strip(),
    }
    
    return candidate


def render_profile(c: dict) -> str:
    """Render candidate profile for prompt context."""
    parts = [
        f"Name: {c.get('name')}",
        f"Current Role: {c.get('current_title')}",
        f"Target Role: {c.get('target_role_title')}",
        f"Location: {c.get('location')} | Timezone: {c.get('timezone')}",
    ]
    
    if c.get("has_visa_issue"):
        parts.append("Note: May require visa sponsorship")
    
    parts.append(f"Experience: {c.get('experience_years')} years")
    parts.append(f"Education: {c.get('education')}")
    
    if c.get("specialization"):
        parts.append(f"Specialization: {c.get('specialization')}")
    
    skills = ", ".join(c.get("skills", []))
    if skills:
        parts.append(f"Skills: {skills}")
    
    highlights = c.get("highlights") or []
    if highlights:
        parts.append("Highlights:\n- " + "\n- ".join(highlights))
    
    pubs = c.get("publications") or []
    if pubs:
        parts.append("Publications:\n- " + "\n- ".join(pubs))
    
    parts.append(f"Job search status: {c.get('openness')}")
    
    jd_excerpt = c.get("job_description_excerpt") or ""
    if jd_excerpt:
        parts.append("\n--- Target Role Description ---\n" + jd_excerpt.strip())
    
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--roles", type=Path, default=DEFAULT_ROLES)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--formatted", type=Path, default=DEFAULT_FORMATTED)
    parser.add_argument("--count", type=int, default=50, help="Number of candidates to generate")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (default: random)")
    parser.add_argument("--edge-case-pct", type=float, default=0.55, help="Fraction of edge case candidates (default raised to create harder evals)")
    parser.add_argument("--replace", action="store_true", help="Replace existing candidates instead of appending")
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
    
    roles = json.loads(args.roles.read_text(encoding="utf-8"))
    
    if args.replace or not args.candidates.exists():
        candidates_existing = []
    else:
        candidates_existing = json.loads(args.candidates.read_text(encoding="utf-8"))

    start_idx = len(candidates_existing) + 1
    
    # Sample roles with content, cycling if needed
    prioritized = [r for r in roles if r.get("content_text")]
    if len(prioritized) < args.count:
        prioritized = roles
    
    sampled = [prioritized[i % len(prioritized)] for i in range(args.count)]

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
    with args.formatted.open("w", encoding="utf-8") as out:
        for c in all_candidates:
            profile_text = render_profile(c)
            record = {"messages": [{"role": "user", "content": profile_text}]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Print summary
    edge_cases = sum(1 for c in new_candidates if c.get("is_edge_case"))
    edge_types = {}
    for c in new_candidates:
        if c.get("edge_case_type"):
            edge_types[c["edge_case_type"]] = edge_types.get(c["edge_case_type"], 0) + 1
    
    domains = {}
    for c in new_candidates:
        d = c.get("domain", "unknown")
        domains[d] = domains.get(d, 0) + 1
    
    print(f"Generated {len(new_candidates)} candidates ({edge_cases} edge cases). Total: {len(all_candidates)}")
    print(f"\nDomain distribution:")
    for d, count in sorted(domains.items(), key=lambda x: -x[1]):
        print(f"  {d}: {count}")
    
    if edge_types:
        print(f"\nEdge case types:")
        for t, count in sorted(edge_types.items(), key=lambda x: -x[1]):
            print(f"  {t}: {count}")
    
    print(f"\nWrote {args.candidates} and {args.formatted}")


if __name__ == "__main__":
    main()
