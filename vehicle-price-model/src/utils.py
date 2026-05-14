"""
Utility functions for Toyota Aqua price prediction project.
"""

def normalize_variant(variant_raw):
    """
    Normalizes the vehicle variant to standard categories.
    - S Grade
    - G Grade
    - Base/Unknown
    """
    if not isinstance(variant_raw, str):
        return "Base/Unknown"
        
    v = variant_raw.strip().upper()
    
    if "S GRADE" in v or v == "S" or " S " in f" {v} " or "S LIMITED" in v:
        return "S Grade"
    elif "G GRADE" in v or v == "G" or " G " in f" {v} " or "G LIMITED" in v or "G LTD" in v:
        return "G Grade"
    else:
        return "Base/Unknown"

def get_year_range(model_year):
    """
    Returns the year range category based on model year.
    """
    try:
        year = int(model_year)
        if year in [2012, 2013, 2014]:
            return "2012-2014"
        elif year in [2015, 2016, 2017]:
            return "2015-2017"
        else:
            return "Other"
    except (ValueError, TypeError):
        return "Other"
