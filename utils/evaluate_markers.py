import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import sys
import re

# ==========================================================
# METRICS IMPLEMENTATION
# ==========================================================

def get_marker_bits(dictionary, marker_id, side=4):
    """
    Generates the image of the ArUco marker and extracts its internal bit matrix (excluding the black border).
    """
    img = aruco.generateImageMarker(dictionary, marker_id, 240)
    cell = img.shape[0] // (side + 2)
    bits = np.zeros((side, side), dtype=int)
    
    for r in range(side):
        for c in range(side):
            y0 = (r + 1) * cell
            y1 = (r + 2) * cell
            x0 = (c + 1) * cell
            x1 = (c + 2) * cell
            roi = img[y0:y1, x0:x1]
            # OpenCV generateImageMarker produces binary values: 0 (black), 255 (white).
            # We average the ROI to determine the bit.
            bits[r, c] = 1 if np.mean(roi) > 127 else 0
            
    return bits

def count_transitions(matrix):
    transitions = 0
    # Rows
    for r in range(4):
        for c in range(3):
            if matrix[r, c] != matrix[r, c+1]:
                transitions += 1
    # Columns
    for c in range(4):
        for r in range(3):
            if matrix[r, c] != matrix[r+1, c]:
                transitions += 1
    return transitions

def max_connected_component(matrix):
    def get_max_cc(val):
        visited = set()
        max_size = 0
        for r in range(4):
            for c in range(4):
                if matrix[r, c] == val and (r, c) not in visited:
                    size = 0
                    queue = [(r, c)]
                    visited.add((r, c))
                    while queue:
                        curr_r, curr_c = queue.pop(0)
                        size += 1
                        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            nr, nc = curr_r + dr, curr_c + dc
                            if 0 <= nr < 4 and 0 <= nc < 4:
                                if matrix[nr, nc] == val and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    queue.append((nr, nc))
                    max_size = max(max_size, size)
        return max_size
    return max(get_max_cc(0), get_max_cc(1))

def count_isolated_cells(matrix):
    isolated = 0
    for r in range(4):
        for c in range(4):
            val = matrix[r, c]
            neighbors = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < 4 and 0 <= nc < 4:
                    neighbors.append(matrix[nr, nc])
            if all(n != val for n in neighbors):
                isolated += 1
    return isolated

def count_chessboard_patterns(matrix):
    patterns = 0
    for r in range(3):
        for c in range(3):
            sub = matrix[r:r+2, c:c+2]
            if (sub[0,0] == sub[1,1] and sub[0,1] == sub[1,0] and sub[0,0] != sub[0,1]):
                patterns += 1
    return patterns

def fully_homogenous_rows_cols(matrix):
    homogenous = 0
    for r in range(4):
        if len(set(matrix[r])) == 1:
            homogenous += 1
    for c in range(4):
        if len(set(matrix[:, c])) == 1:
            homogenous += 1
    return homogenous

# ==========================================================
# PARSING INPUT
# ==========================================================

def parse_ids(input_str):
    """
    Parses a string like '0-100,105,107,110-120' into a sorted list of unique integers.
    """
    ids = set()
    tokens = re.split(r'[,\s]+', input_str.strip())
    for token in tokens:
        if not token:
            continue
        if '-' in token:
            parts = token.split('-')
            if len(parts) == 2:
                try:
                    start = int(parts[0])
                    end = int(parts[1])
                    if start <= end:
                        ids.update(range(start, end + 1))
                    else:
                        ids.update(range(end, start + 1))
                except ValueError:
                    print(f"Warning: could not parse range '{token}'", file=sys.stderr)
            else:
                print(f"Warning: invalid range format '{token}'", file=sys.stderr)
        else:
            try:
                ids.add(int(token))
            except ValueError:
                print(f"Warning: could not parse ID '{token}'", file=sys.stderr)
    return sorted(list(ids))

# ==========================================================
# MAIN EXECUTION
# ==========================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate ArUco markers based on bit matrix quality metrics.")
    parser.add_argument("ids", type=str, help="IDs to evaluate, e.g. '0-10' or '0,1,2,5-8'")
    parser.add_argument("--dict", type=str, default="4X4_250", help="ArUco predefined dictionary (e.g., 4X4_50, 4X4_100, 4X4_250, 4X4_1000). Default is 4X4_250.")
    parser.add_argument("--filter", type=str, choices=['none', 'premium', 'balanced', 'basic'], default='none',
                        help="Apply filter threshold rules: \n"
                             "  premium:  T <= 10, I == 0, Chessboard == 0\n"
                             "  balanced: T <= 11, I <= 1, Chessboard == 0\n"
                             "  basic:    T <= 12, I <= 1, Chessboard == 0")
    
    # Weight settings
    parser.add_argument("--w-cmax", type=float, default=1.0, help="Weight for Max Connected Component (Cmax). Default = 1.0")
    parser.add_argument("--w-homo", type=float, default=1.0, help="Weight for Homogenous Rows/Cols. Default = 1.0")
    parser.add_argument("--w-t", type=float, default=2.0, help="Weight for Transitions (T). Default = 2.0")
    parser.add_argument("--w-i", type=float, default=5.0, help="Weight for Isolated Cells (I). Default = 5.0")
    parser.add_argument("--w-chess", type=float, default=12.0, help="Weight for Chessboard Patterns. Default = 12.0")
    parser.add_argument("--w-ones", type=float, default=1.5, help="Weight for deviation from optimal 7 white cells. Default = 1.5")
    
    args = parser.parse_args()
    
    # Resolve dictionary
    dict_name = f"DICT_{args.dict.upper()}"
    try:
        dict_id = getattr(aruco, dict_name)
    except AttributeError:
        try:
            dict_id = getattr(aruco, args.dict)
        except AttributeError:
            print(f"Error: Unknown dictionary '{args.dict}'.", file=sys.stderr)
            sys.exit(1)
            
    dictionary = aruco.getPredefinedDictionary(dict_id)
    
    # Parse IDs
    target_ids = parse_ids(args.ids)
    if not target_ids:
        print("Error: No valid marker IDs specified.", file=sys.stderr)
        sys.exit(1)
        
    results = []
    filtered_out_count = 0
    
    for marker_id in target_ids:
        try:
            bits = get_marker_bits(dictionary, marker_id)
            cmax = max_connected_component(bits)
            t = count_transitions(bits)
            i = count_isolated_cells(bits)
            chess = count_chessboard_patterns(bits)
            homo = fully_homogenous_rows_cols(bits)
            ones = np.count_nonzero(bits == 1)
            
            # Apply filters
            if args.filter == 'premium':
                if not (t <= 10 and i == 0 and chess == 0):
                    filtered_out_count += 1
                    continue
            elif args.filter == 'balanced':
                if not (t <= 11 and i <= 1 and chess == 0):
                    filtered_out_count += 1
                    continue
            elif args.filter == 'basic':
                if not (t <= 12 and i <= 1 and chess == 0):
                    filtered_out_count += 1
                    continue
            
            # Calculate Q: Q = Cmax + Homo - 2*T - 5*I - 12*Chess - 1.5*abs(Ones - 7)
            ones_penalty = abs(ones - 7)
            q = ((args.w_cmax * cmax) + 
                 (args.w_homo * homo) - 
                 (args.w_t * t) - 
                 (args.w_i * i) - 
                 (args.w_chess * chess) - 
                 (args.w_ones * ones_penalty))
            
            results.append({
                'id': marker_id,
                'q': q,
                'cmax': cmax,
                'homo': homo,
                't': t,
                'i': i,
                'chess': chess,
                'ones': ones
            })
        except Exception as e:
            print(f"Warning: failed to process ID {marker_id}: {e}", file=sys.stderr)
            
    # Sort descending by Q
    results.sort(key=lambda x: x['q'], reverse=True)
    
    # Print results
    print(f"\n--- Marker Quality Evaluation Results (Filter: {args.filter}) ---")
    if filtered_out_count > 0:
        print(f"Filtered out {filtered_out_count} markers that did not meet the '{args.filter}' criteria.")
    print(f"{'ID':<6} | {'Q Score':<8} | {'Cmax':<4} | {'Homo':<4} | {'T':<4} | {'I':<4} | {'Chess':<5} | {'Ones':<4}")
    print("-" * 65)
    for res in results:
        print(f"{res['id']:<6} | {res['q']:<8.2f} | {res['cmax']:<4} | {res['homo']:<4} | {res['t']:<4} | {res['i']:<4} | {res['chess']:<5} | {res['ones']:<4}")
        
    # Output list of IDs ordered by quality descending
    ordered_ids = [str(res['id']) for res in results]
    print("\n--- Ordered IDs (Descending by Quality) ---")
    print(",".join(ordered_ids))

if __name__ == '__main__':
    main()
