from Bio import AlignIO
import os

def get_aln_file(query_path, protein):
    """
    Read a Clustal .aln file and return sequences without leading/trailing gaps.
    """
    aln_path = os.path.join(query_path, f"{protein}.aln")
    alignment = AlignIO.read(aln_path, "clustal")

    # 시작 위치 (leading gap 제거)
    start = next(i for i, c in enumerate(alignment[0].seq) if c != "-")
    # 끝 위치 (trailing gap 제거)
    end = len(alignment[0].seq) - next(i for i, c in enumerate(reversed(alignment[0].seq)) if c != "-")

    return {record.id: record.seq[start:end] for record in alignment}


def make_a3m_file(query_path, protein):
    """
    Convert .aln file to .a3m file.
    """
    aln_path = os.path.join(query_path, f"{protein}.aln")
    a3m_path = os.path.join(query_path, f"{protein}.a3m")

    if not os.path.isfile(aln_path):
        raise FileNotFoundError(f"Alignment file not found: {aln_path}")

    alignment = AlignIO.read(aln_path, "clustal")

    with open(a3m_path, "w") as f:
        for record in alignment:
            f.write(f">{record.id}\n{record.seq}\n")

    return a3m_path


def make_a3m_modify_file(query_path, protein):
    """
    Create a modified .a3m file with query gaps removed.
    """
    a3m_path = os.path.join(query_path, f"{protein}.a3m")
    a3m_mod_path = os.path.join(query_path, f"{protein}_modify.a3m")

    if not os.path.isfile(a3m_path):
        raise FileNotFoundError(f"A3M file not found: {a3m_path}")

    with open(a3m_path, "r") as f:
        lines = f.readlines()

    # query sequence는 두 번째 라인
    query_seq = lines[1].strip()
    gap_indices = {i for i, c in enumerate(query_seq) if c == "-"}

    with open(a3m_mod_path, "w") as f:
        for line in lines:
            if line.startswith(">"):
                f.write(line)
            else:
                seq = "".join([c for i, c in enumerate(line.strip()) if i not in gap_indices])
                f.write(seq + "\n")

    return a3m_mod_path
