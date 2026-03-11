"""
Structured degree requirements and concentration data for Chicago Booth MBA.
Parsed from Degree Requirements.rtf and Concentration Requirements.rtf.
"""

# ============================================================
# DEGREE REQUIREMENTS (2025-26 Curriculum)
# ============================================================

DEGREE_REQUIREMENTS = {
    "total_units": 2000,
    "min_booth_units": 1400,

    "leadership": {
        "name": "LEAD",
        "units": 0,
        "courses": ["31001"],
        "description": "Leadership Exploration and Development (0 credit)",
    },

    "qwe": {
        "name": "Qualified Work Experience",
        "units": 0,
        "description": "Full-Time MBA only, matriculating Autumn 2020+",
    },

    "foundations": {
        "units_required": 300,  # 100 per area, 3 areas
        "areas": {
            "Financial Accounting": {
                "units": 100,
                "basic": ["30000"],
                "substitutes": ["30116", "30120", "30122", "30130", "30131"],
            },
            "Microeconomics": {
                "units": 100,
                "basic": ["33001", "33002", "33101"],
                "substitutes": ["33230", "ECON 30100", "ECON 30200"],
            },
            "Statistics": {
                "units": 100,
                "basic": ["41000", "41100"],
                "substitutes": [
                    "41202", "41203", "41204", "41206", "41207",
                    "41215", "41201", "41301", "41305", "41813",
                    "41814", "41901", "41902", "41903", "41910", "41916",
                ],
            },
        },
    },

    "flmbe": {
        "name": "Functions, Leadership and Management, and the Business Environment",
        "units_required": 700,  # 7 of 8 areas, 100 each
        "areas_required": 7,
        "areas_total": 8,
        "categories": {
            "Functions": {
                "Finance": {
                    "units": 100,
                    "basic": ["35000", "35001", "35200"],
                    "substitutes": [
                        "34101", "34901", "34902", "34903", "34904",
                        "35100", "35120", "35130", "35150", "35201", "35210", "35214",
                    ],
                },
                "Marketing": {
                    "units": 100,
                    "basic": ["37000", "37110"],
                    "substitutes": [
                        "37101", "37103", "37105", "37106", "37107",
                        "37200", "37201", "37202", "37208", "37209",
                        "37301", "37304", "37703", "37704",
                    ],
                },
                "Operations": {
                    "units": 100,
                    "basic": ["40000"],
                    "substitutes": ["40101", "40108", "40110"],
                },
                "Strategy": {
                    "units": 100,
                    "basic": ["42001"],
                    "substitutes": ["39001", "39101", "42116", "42135", "42715"],
                },
            },
            "Leadership & Management": {
                "Decisions": {
                    "units": 100,
                    "basic": ["30005", "30001", "36106", "38002", "38120"],
                    "substitutes": ["36109"],
                },
                "People": {
                    "units": 100,
                    "basic": ["33032", "38001", "38003", "39002"],
                    "substitutes": ["31403", "38122"],
                },
            },
            "Business Environment": {
                "Economy": {
                    "units": 100,
                    "basic": ["33050", "33040", "33112"],
                    "substitutes": ["33401", "33403", "33501", "33502", "33503", "33520"],
                },
                "Society": {
                    "units": 100,
                    "basic": ["33305", "33471", "37212", "38119"],
                    "substitutes": [
                        "30133", "33251", "34113", "34117",
                        "38115", "38126", "42136", "42201",
                    ],
                },
            },
        },
    },

    "electives": {
        "units_required": 1000,
        "description": "Choose 1000 units of electives",
    },
}


# ============================================================
# CONCENTRATION REQUIREMENTS
# ============================================================

CONCENTRATIONS = {
    "Accounting": {
        "units_required": 400,
        "courses": [
            "30000", "30005", "30001", "30116", "30118", "30120", "30121",
            "30122", "30130", "30131", "30132", "30133", "30135",
            "30830", "30831", "30835", "30840",
        ],
    },
    "Applied Artificial Intelligence": {
        "units_required": 300,
        "courses": [
            "30135", "32210", "32810", "35137",
            "43100", "32200", "43110", "43120", "43130", "43800", "43950",
        ],
        "notes": "Only 100 units can count toward both Applied AI and Business Analytics.",
    },
    "Behavioral Science": {
        "units_required": 400,
        "courses": [
            "31403", "31702", "38001", "38002", "38003", "38101", "38102",
            "38103", "38105", "38107", "38115", "38116", "38118", "38119",
            "38120", "38122", "38123", "38126", "38820", "38821", "38825",
            "38870", "38865", "38886", "39002",
        ],
    },
    "Business Analytics": {
        "units_required": 500,
        "structure": {
            "data_science": {
                "units": 100,
                "courses": ["41215", "41201", "41204"],
            },
            "decision_models": {
                "units": 100,
                "courses": ["36106", "36109"],
            },
            "electives": {
                "units": 300,
                "courses": [
                    "32100", "32120", "32130", "32810", "35126", "35137",
                    "36109", "37103", "37105", "37107", "37202", "37802",
                    "40108", "40206", "40205", "40721", "41201", "41202",
                    "41203", "41204", "41215", "41301", "41305",
                    "43100", "32200", "41814",
                ],
                "max_per_area": 200,
            },
        },
        "notes": "Only 100 units can count toward both Applied AI and Business Analytics.",
    },
    "Business, Society and Sustainability": {
        "units_required": 400,
        "courses": [
            "30133", "31425", "33251", "33305", "33471", "34113", "34115",
            "34117", "35225", "37212", "38115", "38119", "38122", "38128",
            "42129", "42136", "42125", "42708", "42710", "42711", "42712", "43130",
        ],
    },
    "Econometrics and Statistics": {
        "units_required": 300,
        "courses": [
            "41000", "41100", "41202", "41203", "41204", "41206", "41813",
            "41207", "41215", "41201", "41301", "41305", "41814",
            "41901", "41902", "41903", "41910",
        ],
    },
    "Economics": {
        "units_required": 400,
        "courses": [
            "33032", "33050", "33040", "33101", "33112", "33230", "33251",
            "33301", "33305", "33320", "33350", "33401", "33403", "33454",
            "33501", "33502", "33503", "33520", "33882", "38120",
            "42001", "42116", "42135",
        ],
        "notes": "33001 and 33002 do not qualify toward the concentration.",
    },
    "Entrepreneurship": {
        "units_required": 300,
        "courses": [
            "30118", "30121", "31401", "31402", "31403", "34101", "34102",
            "34103", "34104", "34106", "34108", "34111", "34113", "34115",
            "34117", "34205", "34206", "34208", "34210", "34211", "34214",
            "34215", "34219", "34220", "34302", "34305", "34308", "34702",
            "34704", "34705", "34709", "34715", "34815", "34816", "34820",
            "34825", "34826", "34880", "34882", "34887", "35123", "35213",
            "35823", "36110", "37201", "37200", "37301", "37703", "39101",
            "40110", "41206", "41813", "41301", "41305", "42705", "42711",
            "42820", "42830", "43120",
        ],
        "notes": "Only 100 units can count toward both Entrepreneurship and Strategic Management.",
    },
    "Finance": {
        "units_required": 400,
        "structure": {
            "asset_pricing": {
                "units": 100,
                "courses": [
                    "34901", "34902", "35000", "35100", "35101", "35120",
                    "35121", "35126", "35130", "35132", "35901", "35908",
                ],
            },
            "corporate_finance": {
                "units": 100,
                "courses": [
                    "34101", "34903", "34904", "34905", "34906",
                    "35200", "35201", "35202", "35210", "35213", "35214", "35215",
                ],
            },
            "additional": {
                "units": 200,
                "courses": [
                    "30130", "30131", "34101", "34901", "34902", "34903", "34904",
                    "35100", "35101", "35118", "35120", "35121", "35123", "35126",
                    "35130", "35131", "35136", "35135", "35137", "35139", "35141",
                    "35144", "35124", "35146", "35145", "35150", "35155",
                    "35201", "35202", "35207", "35210", "35213", "35214", "35215",
                    "35218", "35219", "35222", "35220", "35225", "35817", "35823",
                    "35824", "35830", "35889", "35901", "35907", "35908", "35916",
                    "41203",
                ],
            },
        },
    },
    "Analytic Finance": {
        "units_required": 600,
        "courses": [
            "34101", "34901", "34902", "34903", "34904", "34905", "34906",
            "35100", "35118", "35120", "35121", "35123", "35126", "35130",
            "35136", "35137", "35139", "35141", "35150", "35155",
            "35200", "35201", "35202", "35207", "35210", "35213", "35214",
            "35215", "35218", "35219", "35901", "35907", "35908", "35916",
            "41203",
        ],
        "notes": "Must also satisfy Finance concentration requirements.",
    },
    "General Management": {
        "units_required": 1100,
        "description": "Complete all 8 FLMBE areas (800 units) + 300 units from Strategic Management and/or Behavioral Science concentration lists.",
    },
    "Healthcare": {
        "units_required": 400,
        "structure": {
            "core": {
                "units": 100,
                "courses": ["33350", "40206", "40205"],
            },
            "electives": {
                "units": "200-300",
                "courses": [
                    "33351", "33352", "33353", "34205", "34210",
                    "35118", "42300", "42310",
                ],
            },
            "experiential": {
                "units": "up to 100",
                "courses": [
                    "34104", "34115", "34702", "34705", "34709",
                    "35817", "37201", "37703", "40721", "42709", "42710",
                ],
                "notes": "Must be pre-approved by petition with a healthcare project.",
            },
        },
    },
    "International Business": {
        "units_required": 300,
        "courses": [
            "30131", "33501", "33502", "33503", "33520", "35210", "35213", "35219",
        ],
        "notes": "At least one must be 33501 or 33502.",
    },
    "Marketing Management": {
        "units_required": 400,
        "structure": {
            "required": {
                "units": 100,
                "courses": ["37000", "37110"],
            },
            "electives": {
                "units": 300,
                "courses": [
                    "37101", "37103", "37105", "37107", "37110",
                    "37200", "37201", "37202", "37208", "37209", "37212",
                    "37215", "37301", "37703", "37810", "37816", "37820",
                    "37882", "37902", "41301", "41305",
                ],
            },
        },
    },
    "Operations Management": {
        "units_required": 300,
        "courses": [
            "36106", "36109", "40000", "40101", "40108", "40110",
            "40111", "40206", "40205", "40721", "40812",
        ],
    },
    "Strategic Management": {
        "units_required": 400,
        "courses": [
            "32200", "33230", "33503", "34102", "34103", "34106", "34108",
            "34117", "34220", "34816", "37201", "37200", "39001", "39101",
            "42001", "42108", "42116", "42119", "42121", "42123", "42133",
            "42124", "42126", "42127", "42135", "42136", "42125",
            "42705", "42708", "42709", "42710", "42711", "42715", "42726",
            "42813", "42819", "42820", "42830",
        ],
        "notes": "Only 100 units can count toward both Entrepreneurship and Strategic Management.",
    },
}


def get_all_requirement_courses():
    """Get a set of all course codes that appear in any requirement."""
    codes = set()

    # Foundations
    for area in DEGREE_REQUIREMENTS["foundations"]["areas"].values():
        codes.update(area["basic"])
        codes.update(area["substitutes"])

    # FLMBE
    for category in DEGREE_REQUIREMENTS["flmbe"]["categories"].values():
        for area in category.values():
            codes.update(area["basic"])
            codes.update(area["substitutes"])

    return codes


def get_all_concentration_courses():
    """Get a mapping of concentration name -> set of course codes."""
    result = {}
    for name, conc in CONCENTRATIONS.items():
        courses = set()
        if "courses" in conc:
            courses.update(conc["courses"])
        if "structure" in conc:
            for sub in conc["structure"].values():
                if "courses" in sub:
                    courses.update(sub["courses"])
        result[name] = courses
    return result


def check_course_requirements(course_code):
    """
    Check what requirements a course fulfills.
    Returns a dict with foundation, flmbe, and concentration info.
    """
    result = {
        "foundations": [],
        "flmbe": [],
        "concentrations": [],
    }

    code = course_code.split("-")[0].strip()

    # Check foundations
    for area_name, area in DEGREE_REQUIREMENTS["foundations"]["areas"].items():
        if code in area["basic"] or code in area["substitutes"]:
            result["foundations"].append(area_name)

    # Check FLMBE
    for cat_name, category in DEGREE_REQUIREMENTS["flmbe"]["categories"].items():
        for area_name, area in category.items():
            if code in area["basic"] or code in area["substitutes"]:
                result["flmbe"].append(f"{cat_name} > {area_name}")

    # Check concentrations
    conc_courses = get_all_concentration_courses()
    for conc_name, courses in conc_courses.items():
        if code in courses:
            result["concentrations"].append(conc_name)

    return result


if __name__ == "__main__":
    # Verify data
    req_courses = get_all_requirement_courses()
    conc_courses = get_all_concentration_courses()

    print(f"Unique courses in degree requirements: {len(req_courses)}")
    print(f"Concentrations: {len(CONCENTRATIONS)}")
    for name, courses in conc_courses.items():
        print(f"  {name}: {len(courses)} courses, {CONCENTRATIONS[name]['units_required']} units required")

    # Example: check what 35200 (Corporation Finance) fulfills
    print("\n=== Example: 35200 (Corporation Finance) ===")
    info = check_course_requirements("35200")
    print(f"  Foundations: {info['foundations']}")
    print(f"  FLMBE: {info['flmbe']}")
    print(f"  Concentrations: {info['concentrations']}")

    print("\n=== Example: 38120 (Behavioral Economics) ===")
    info = check_course_requirements("38120")
    print(f"  Foundations: {info['foundations']}")
    print(f"  FLMBE: {info['flmbe']}")
    print(f"  Concentrations: {info['concentrations']}")
