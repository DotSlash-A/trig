# services/statistics_services.py
import math
from typing import List, Dict, Any, Tuple, Union, Optional
from collections import Counter

# --- Ungrouped Data ---
def calculate_mean_ungrouped(data: List[float]) -> float:
    if not data:
        raise ValueError("Data list cannot be empty.")
    return sum(data) / len(data)

def calculate_median_ungrouped(data: List[float]) -> float:
    if not data:
        raise ValueError("Data list cannot be empty.")
    sorted_data = sorted(data)
    n = len(sorted_data)
    mid = n // 2
    if n % 2 == 0: # Even number of observations
        return (sorted_data[mid - 1] + sorted_data[mid]) / 2
    else: # Odd number of observations
        return sorted_data[mid]

def calculate_mode_ungrouped(data: List[Union[float, int, str]]) -> List[Union[float, int, str]]:
    if not data:
        raise ValueError("Data list cannot be empty.")
    counts = Counter(data)
    max_freq = 0
    # Find the maximum frequency
    for item in counts:
        if counts[item] > max_freq:
            max_freq = counts[item]
    
    if max_freq == 1 and len(set(data)) == len(data): # All items unique or appear once
        return ["No mode" if len(data) > 1 else data[0]] # If only one item, it's the mode

    modes = [item for item, count in counts.items() if count == max_freq]
    
    # If all items have the same frequency and it's > 1 (e.g., [1,1,2,2,3,3]), no distinct mode
    if len(modes) == len(set(data)) and max_freq > 1 and len(set(data)) > 1:
        return ["No distinct mode (all items have same highest frequency)"]
        
    return sorted(modes) # Return sorted list of modes (can be multimodal)

# --- Grouped Data ---
# Helper structure for grouped data items
class GroupedDataClass:
    def __init__(self, lower_limit: float, upper_limit: float, frequency: int):
        if lower_limit >= upper_limit:
            raise ValueError(f"Lower limit ({lower_limit}) must be less than upper limit ({upper_limit}).")
        if frequency < 0:
            raise ValueError("Frequency cannot be negative.")
        self.lower_limit = lower_limit
        self.upper_limit = upper_limit
        self.frequency = frequency
        self.class_mark = (lower_limit + upper_limit) / 2
        self.class_size = upper_limit - lower_limit

def _prepare_grouped_data_table(grouped_data_input: List[Dict[str, Union[float, int]]]) -> Tuple[List[GroupedDataClass], List[Dict[str, Any]]]:
    """Validates and processes input into GroupedDataClass objects and a summary table."""
    if not grouped_data_input:
        raise ValueError("Grouped data input cannot be empty.")
    
    data_classes: List[GroupedDataClass] = []
    for item in grouped_data_input:
        try:
            # Handle both "class_interval" string and separate limits
            if "class_interval" in item and isinstance(item["class_interval"], str):
                parts = item["class_interval"].split('-')
                if len(parts) != 2:
                    raise ValueError(f"Invalid class_interval format: {item['class_interval']}. Expected 'lower-upper'.")
                lower = float(parts[0].strip())
                upper = float(parts[1].strip())
            elif "lower_limit" in item and "upper_limit" in item:
                lower = float(item["lower_limit"])
                upper = float(item["upper_limit"])
            else:
                raise ValueError("Each item must have 'class_interval' or 'lower_limit' and 'upper_limit'.")

            freq = int(item["frequency"])
            data_classes.append(GroupedDataClass(lower, upper, freq))
        except (ValueError, TypeError, KeyError) as e:
            raise ValueError(f"Invalid data format in grouped_data_input: {item}. Error: {e}")

    # Sort by lower limit to ensure order
    data_classes.sort(key=lambda x: x.lower_limit)

    # Basic validation for continuous classes (optional but good)
    for i in range(len(data_classes) - 1):
        if abs(data_classes[i].upper_limit - data_classes[i+1].lower_limit) > 1e-6 : # Tolerance for float
            # Allow for non-continuous if user intends, but flag it or make it strict
            # For mean/median/mode calculations, continuity is usually assumed.
            print(f"Warning: Classes may not be continuous between {data_classes[i].upper_limit} and {data_classes[i+1].lower_limit}")


    # Prepare a summary table (list of dicts) for response, including cumulative frequency
    table_summary = []
    cumulative_frequency = 0
    for dc in data_classes:
        cumulative_frequency += dc.frequency
        table_summary.append({
            "class_interval": f"{dc.lower_limit}-{dc.upper_limit}",
            "frequency (f_i)": dc.frequency,
            "class_mark (x_i)": dc.class_mark,
            "cumulative_frequency (cf)": cumulative_frequency
        })
    return data_classes, table_summary


def calculate_mean_grouped_direct(data_classes: List[GroupedDataClass]) -> Tuple[float, List[Dict[str, Any]]]:
    if not data_classes:
        raise ValueError("Data classes list cannot be empty.")
    
    sum_fi_xi = 0
    sum_fi = 0
    calculation_steps = []

    for dc in data_classes:
        fi_xi = dc.frequency * dc.class_mark
        sum_fi_xi += fi_xi
        sum_fi += dc.frequency
        calculation_steps.append({
            "class_interval": f"{dc.lower_limit}-{dc.upper_limit}",
            "f_i": dc.frequency,
            "x_i": dc.class_mark,
            "f_i*x_i": fi_xi
        })
    
    if sum_fi == 0:
        raise ValueError("Total frequency (sum_fi) is zero, cannot calculate mean.")
        
    mean = sum_fi_xi / sum_fi
    summary_table = {
        "method": "Direct Method",
        "steps_table": calculation_steps,
        "sum_fi_xi": sum_fi_xi,
        "sum_fi": sum_fi,
        "mean": mean,
        "formula": "Mean = Σ(fᵢxᵢ) / Σfᵢ"
    }
    return mean, summary_table

def calculate_mean_grouped_assumed_mean(data_classes: List[GroupedDataClass], assumed_mean_val: Optional[float] = None) -> Tuple[float, List[Dict[str, Any]]]:
    if not data_classes:
        raise ValueError("Data classes list cannot be empty.")

    if assumed_mean_val is None:
        # Choose assumed mean, often the class mark of the middle class or class with highest frequency
        mid_index = len(data_classes) // 2
        a = data_classes[mid_index].class_mark
    else:
        a = assumed_mean_val
        
    sum_fi_di = 0
    sum_fi = 0
    calculation_steps = []

    for dc in data_classes:
        di = dc.class_mark - a
        fi_di = dc.frequency * di
        sum_fi_di += fi_di
        sum_fi += dc.frequency
        calculation_steps.append({
            "class_interval": f"{dc.lower_limit}-{dc.upper_limit}",
            "f_i": dc.frequency,
            "x_i": dc.class_mark,
            "d_i (x_i - a)": di,
            "f_i*d_i": fi_di
        })

    if sum_fi == 0:
        raise ValueError("Total frequency (sum_fi) is zero, cannot calculate mean.")
        
    mean = a + (sum_fi_di / sum_fi)
    summary_table = {
        "method": "Assumed Mean Method",
        "assumed_mean (a)": a,
        "steps_table": calculation_steps,
        "sum_fi_di": sum_fi_di,
        "sum_fi": sum_fi,
        "mean": mean,
        "formula": "Mean = a + Σ(fᵢdᵢ) / Σfᵢ"
    }
    return mean, summary_table

def calculate_mean_grouped_step_deviation(data_classes: List[GroupedDataClass], assumed_mean_val: Optional[float] = None, class_height_h: Optional[float] = None) -> Tuple[float, List[Dict[str, Any]]]:
    if not data_classes:
        raise ValueError("Data classes list cannot be empty.")

    # Determine class height 'h' if not provided (assuming uniform class width)
    if class_height_h is None:
        if len(set(dc.class_size for dc in data_classes)) != 1:
            raise ValueError("Class sizes are not uniform. Step-deviation method requires a common class height 'h'. Please provide 'h'.")
        h = data_classes[0].class_size
    else:
        h = class_height_h
    
    if h <= 1e-9: # h is zero or too small
        raise ValueError("Class height 'h' must be positive and non-zero.")

    if assumed_mean_val is None:
        mid_index = len(data_classes) // 2
        a = data_classes[mid_index].class_mark
    else:
        a = assumed_mean_val
        
    sum_fi_ui = 0
    sum_fi = 0
    calculation_steps = []

    for dc in data_classes:
        di = dc.class_mark - a
        ui = di / h 
        # Check if ui is reasonably close to an integer, common for this method
        if abs(ui - round(ui)) > 1e-6: # If not close to integer, step deviation might be misapplied or 'a' chosen poorly
            print(f"Warning: u_i value {ui} for class {dc.lower_limit}-{dc.upper_limit} is not close to an integer. Check assumed mean 'a' and class height 'h'.")
        # ui = round(ui) # Often rounded in textbook examples, but can lead to precision loss. Let's use actual.

        fi_ui = dc.frequency * ui
        sum_fi_ui += fi_ui
        sum_fi += dc.frequency
        calculation_steps.append({
            "class_interval": f"{dc.lower_limit}-{dc.upper_limit}",
            "f_i": dc.frequency,
            "x_i": dc.class_mark,
            "d_i (x_i - a)": di,
            "u_i (d_i/h)": ui,
            "f_i*u_i": fi_ui
        })
        
    if sum_fi == 0:
        raise ValueError("Total frequency (sum_fi) is zero, cannot calculate mean.")

    mean = a + (sum_fi_ui / sum_fi) * h
    summary_table = {
        "method": "Step-Deviation Method",
        "assumed_mean (a)": a,
        "class_height (h)": h,
        "steps_table": calculation_steps,
        "sum_fi_ui": sum_fi_ui,
        "sum_fi": sum_fi,
        "mean": mean,
        "formula": "Mean = a + (Σ(fᵢuᵢ) / Σfᵢ) * h"
    }
    return mean, summary_table


def calculate_median_grouped(data_classes: List[GroupedDataClass], full_table: List[Dict[str,Any]]) -> Tuple[float, Dict[str, Any]]:
    if not data_classes:
        raise ValueError("Data classes list cannot be empty.")

    total_frequency_n = sum(dc.frequency for dc in data_classes)
    if total_frequency_n == 0:
        raise ValueError("Total frequency is zero, cannot calculate median.")

    n_by_2 = total_frequency_n / 2
    
    # Find median class (class whose cumulative frequency is >= n/2)
    median_class_index = -1
    # cumulative frequencies are in full_table
    for i, row in enumerate(full_table):
        if row["cumulative_frequency (cf)"] >= n_by_2:
            median_class_index = i
            break
            
    if median_class_index == -1: # Should not happen if data is valid
        raise ValueError("Could not determine median class.")

    median_class_info = data_classes[median_class_index]
    l = median_class_info.lower_limit
    f = median_class_info.frequency
    h = median_class_info.class_size

    if f == 0: # Frequency of median class is zero
        # This is an edge case. If median class has 0 freq, it implies data is sparse or incorrectly grouped.
        # Median would technically fall on boundary or need re-evaluation of classes.
        # For simplicity, raise error or indicate ambiguity.
        raise ValueError(f"Frequency of the identified median class ({l}-{median_class_info.upper_limit}) is zero. Median is ill-defined or data needs review.")


    # Cumulative frequency of the class PRECEDING the median class
    cf = 0
    if median_class_index > 0:
        cf = full_table[median_class_index - 1]["cumulative_frequency (cf)"]
    
    median = l + ((n_by_2 - cf) / f) * h
    
    calculation_details = {
        "total_frequency (N)": total_frequency_n,
        "N/2": n_by_2,
        "median_class_interval": f"{median_class_info.lower_limit}-{median_class_info.upper_limit}",
        "lower_limit_median_class (l)": l,
        "cumulative_freq_preceding_class (cf)": cf,
        "frequency_median_class (f)": f,
        "class_size (h)": h,
        "median": median,
        "formula": "Median = l + [ (N/2 - cf) / f ] * h"
    }
    return median, calculation_details

def calculate_mode_grouped(data_classes: List[GroupedDataClass]) -> Tuple[float, Dict[str, Any]]:
    if not data_classes:
        raise ValueError("Data classes list cannot be empty.")

    # Find modal class (class with the highest frequency)
    max_freq = -1
    modal_class_index = -1
    for i, dc in enumerate(data_classes):
        if dc.frequency > max_freq:
            max_freq = dc.frequency
            modal_class_index = i
        # If multiple classes have same max_freq, standard formula usually takes the first one.
        # Or indicates multimodal if they are not adjacent. For simplicity, take first.
    
    if modal_class_index == -1 or max_freq == 0: # No frequencies or all frequencies are zero
        raise ValueError("Cannot determine modal class (e.g., all frequencies are zero or no data).")

    modal_class_info = data_classes[modal_class_index]
    l = modal_class_info.lower_limit
    f1 = modal_class_info.frequency # Frequency of modal class
    h = modal_class_info.class_size

    # Frequency of class preceding modal class (f0)
    f0 = 0
    if modal_class_index > 0:
        f0 = data_classes[modal_class_index - 1].frequency
    
    # Frequency of class succeeding modal class (f2)
    f2 = 0
    if modal_class_index < len(data_classes) - 1:
        f2 = data_classes[modal_class_index + 1].frequency
        
    # Check denominator: 2f1 - f0 - f2
    denominator = 2 * f1 - f0 - f2
    if abs(denominator) < 1e-9: # Denominator is zero
        # This happens in specific distributions (e.g., uniform part, or f1 is not "peaked" enough)
        # Mode is ill-defined by this formula or may need alternative interpretation (e.g. midpoint if f0=f1=f2)
        # For now, indicate this issue. A common approach is to take mode as l + h/2 if f0=f2 and f1 > f0.
        # Or use empirical relation: Mode = 3*Median - 2*Mean (but we want direct calc)
        
        # Handle edge cases for mode based on NCERT/common practices:
        # If f0 and f2 are both 0, and f1 is the only freq, mode is often midpoint of modal class.
        # Or if the modal class is the first class, f0=0. If it's the last, f2=0.
        # The formula itself handles f0=0 or f2=0. The issue is if 2f1-f0-f2 = 0.
        # This can happen if f1=f0 and f2=0, or f1=f2 and f0=0, or if f1 = (f0+f2)/2.
        raise ValueError(f"Denominator (2f₁ - f₀ - f₂) is zero for modal class {modal_class_info.lower_limit}-{modal_class_info.upper_limit}. Mode calculation by this formula is problematic. Check data distribution.")

    mode = l + ((f1 - f0) / denominator) * h
    
    # Ensure mode is within the modal class boundaries (or very close due to float precision)
    if mode < l - 1e-6 or mode > modal_class_info.upper_limit + 1e-6:
        print(f"Warning: Calculated mode {mode} is outside the modal class interval [{l}, {modal_class_info.upper_limit}]. This can happen in skewed distributions or with wide classes.")
        # It's possible, but usually indicates the formula might not be best fit for the shape.

    calculation_details = {
        "modal_class_interval": f"{modal_class_info.lower_limit}-{modal_class_info.upper_limit}",
        "lower_limit_modal_class (l)": l,
        "frequency_modal_class (f₁)": f1,
        "frequency_preceding_class (f₀)": f0,
        "frequency_succeeding_class (f₂)": f2,
        "class_size (h)": h,
        "mode": mode,
        "formula": "Mode = l + [ (f₁ - f₀) / (2f₁ - f₀ - f₂) ] * h"
    }
    return mode, calculation_details

# --- Ogive Data Generation ---
# (Actual plotting is client-side. This service provides data points for ogives)
def generate_ogive_data(full_table: List[Dict[str,Any]], data_classes: List[GroupedDataClass]) -> Dict[str, List[Dict[str, float]]]:
    """
    Generates data points for plotting 'Less than' and 'More than' Ogives.
    full_table should contain 'cumulative_frequency (cf)'.
    data_classes for limits.
    """
    if not full_table or not data_classes:
        raise ValueError("Input tables cannot be empty.")

    less_than_ogive = []
    # For less than ogive, points are (Upper Limit, Cumulative Frequency)
    # Start with (Lower limit of first class, 0)
    less_than_ogive.append({"x_upper_limit": data_classes[0].lower_limit, "y_cumulative_frequency": 0})
    for i, row in enumerate(full_table):
        less_than_ogive.append({
            "x_upper_limit": data_classes[i].upper_limit,
            "y_cumulative_frequency": row["cumulative_frequency (cf)"]
        })

    more_than_ogive = []
    # For more than ogive, points are (Lower Limit, N - cf_preceding) or (Lower Limit, sum of freqs from this class onwards)
    # Start with (Lower limit of first class, Total Frequency N)
    total_frequency_n = full_table[-1]["cumulative_frequency (cf)"]
    
    current_sum_from_here = total_frequency_n
    for i, dc in enumerate(data_classes):
        more_than_ogive.append({
            "x_lower_limit": dc.lower_limit,
            "y_cumulative_frequency_from_here": current_sum_from_here
        })
        current_sum_from_here -= dc.frequency
    # Add last point: (Upper limit of last class, 0)
    more_than_ogive.append({"x_lower_limit": data_classes[-1].upper_limit, "y_cumulative_frequency_from_here": 0})
    
    return {
        "less_than_ogive_points": less_than_ogive,
        "more_than_ogive_points": more_than_ogive
    }