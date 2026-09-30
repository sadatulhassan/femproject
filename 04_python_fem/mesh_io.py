import numpy as np


def read_ansys_cdb(filename):

    with open(filename, "r") as f:
        lines = f.readlines()

    # -------------------------------------------------
    # FIND NBLOCK AND EBLOCK
    # -------------------------------------------------

    nblock_start = None
    eblock_start = None

    for i, line in enumerate(lines):

        if line.startswith("NBLOCK"):
            nblock_start = i

        elif line.startswith("EBLOCK"):
            eblock_start = i
            break

    if nblock_start is None:
        raise ValueError("NBLOCK not found.")

    if eblock_start is None:
        raise ValueError("EBLOCK not found.")

    # -------------------------------------------------
    # READ NODES
    # -------------------------------------------------

    nodes = {}

    # Skip:
    # NBLOCK line
    # format line

    i = nblock_start + 2

    while i < eblock_start:

        line = lines[i]

        if line.strip() == "":
            i += 1
            continue

        try:

            # ANSYS format:
            # (3i9,6e21.13e3)

            node_id = int(line[0:9])
# Read coordinates.
# Some ANSYS CDB lines omit trailing zero coordinates,
# so missing coordinates are treated as zero.

            x = float(line[27:48])

            if len(line) >= 69:
                 y = float(line[48:69])
            else:
                 y = 0.0

            if len(line) >= 90:
                 z = float(line[69:90])
            else:
                 z = 0.0
            # Your model lies in X-Z plane.
            # Therefore use X and Z for 2-D FEM.

            nodes[node_id] = [x, z]

        except ValueError:
            pass

        i += 1

    # -------------------------------------------------
    # READ ELEMENTS
    # -------------------------------------------------

    elements = []

    # Skip:
    # EBLOCK line
    # format line

    i = eblock_start + 2

    while i < len(lines):

        line = lines[i]

        if line.strip() == "-1":
            break

        if line.strip() == "":
            i += 1
            continue

        try:

            # ANSYS format:
            # (19i9)

            values = []

            for j in range(0, 171, 9):

                field = line[j:j + 9]

                if field.strip():
                    values.append(int(field))

            # Element record:
            #
            # index 10 = element ID
            # index 11 = node 1
            # index 12 = node 2
            # index 13 = node 3
            # index 14 = node 4

            if len(values) >= 15:

                element_id = values[10]

                n1 = values[11]
                n2 = values[12]
                n3 = values[13]
                n4 = values[14]

                elements.append([
                    n1,
                    n2,
                    n3,
                    n4
                ])

        except ValueError:
            pass

        i += 1

    # -------------------------------------------------
    # CONVERT NODE IDS TO ARRAY INDICES
    # -------------------------------------------------

    node_ids = sorted(nodes.keys())

    node_array = np.array(
        [nodes[node_id] for node_id in node_ids],
        dtype=float
    )

    node_id_to_index = {
        node_id: index
        for index, node_id in enumerate(node_ids)
    }

    # Convert ANSYS node numbers to Python array indices

    element_array = []

    for element in elements:

        try:

            element_array.append([
                node_id_to_index[element[0]],
                node_id_to_index[element[1]],
                node_id_to_index[element[2]],
                node_id_to_index[element[3]]
            ])

        except KeyError:
            print("Warning: element contains unknown node:", element)

    element_array = np.array(
        element_array,
        dtype=int
    )

    return node_array, element_array

def classify_elements(elements):
    """
    Separate normal 4-node elements from
    degenerate elements with a repeated node.
    """

    q4_elements = []
    degenerate_elements = []

    for i, element in enumerate(elements):

        # Check whether all four nodes are different
        if len(set(element)) == 4:
            q4_elements.append(element)
        else:
            degenerate_elements.append((i, element))

    return (
        np.array(q4_elements, dtype=int),
        degenerate_elements
    )

def correct_element_orientation(nodes, elements):
    """
    Make Q4 elements counter-clockwise.
    """

    corrected_elements = []

    for element in elements:

        coords = nodes[element]

        area = 0.0

        for i in range(4):

            x1, y1 = coords[i]
            x2, y2 = coords[(i + 1) % 4]

            area += x1 * y2 - x2 * y1

        area = 0.5 * area

        if area < 0:
            element = element[::-1]

        corrected_elements.append(element)

    return np.array(corrected_elements, dtype=int)
# =====================================================
# TEST
# =====================================================
if __name__ == "__main__":

    nodes, elements = read_ansys_cdb("my_mesh.cdb")

    q4_elements, degenerate_elements = classify_elements(elements)

    print("========================================")
    print("          ANSYS CDB MESH")
    print("========================================")

    print(f"Number of nodes       : {len(nodes)}")
    print(f"Total elements        : {len(elements)}")
    print(f"Q4 elements           : {len(q4_elements)}")
    print(f"Degenerate elements   : {len(degenerate_elements)}")

    print("\nFirst 5 nodes:")
    print(nodes[:5])

    print("\nFirst 5 Q4 elements:")
    print(q4_elements[:5])

    print("\nDegenerate elements:")

    for index, element in degenerate_elements:
        print(f"Element {index + 1}: {element}")

    print("========================================")