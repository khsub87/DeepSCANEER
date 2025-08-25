from .__init__ import *
import os, subprocess
from io import StringIO
from Bio import SeqIO
from Bio import ExPASy, SwissProt

AA_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
blast_pth = "/home/khsub/anaconda3/pkgs/blast-2.5.0-hc0b0e79_3/bin/"

def retrieve_fasta(accession, db='/home/khsub/PE_proj/code/dhpylib/db/uniprot_sprot_0221.fasta', expasy = True):
	try:
		blastcmd_db = os.path.join(lib_path, 'db', db)
		blastcmd_exe = os.path.join(blast_pth, get_executable_arch('blastdbcmd')) 
		sf = StringIO() 
		sf.write(subprocess.getoutput("%s -entry %s -db %s" %\
				  (blastcmd_exe, accession, blastcmd_db)))
		sf.seek(0)
		return str(SeqIO.read(sf, "fasta").seq)
	# 
	except:
		if db.find('uniprot') != -1:
			if expasy:
				print("From ExPASy")
				handle = ExPASy.get_sprot_raw(accession)
				record = SwissProt.read(handle)
				return record.sequence
			else:
				return None

def retrieve_latest_unp(accession):
	handle = ExPASy.get_sprot_raw(accession)
	record = SwissProt.read(handle)
	return record.sequence

def make_fastas_from_file(id_file, fasta_output_file, db='/home/kimdh4962/dhpylib/db/uniprot_sprot_0221.fasta'):
	f = open(id_file)
	fo = open(fasta_output_file, 'w')
	for line in f:
		id = line.strip()
		seq = retrieve_fasta(id, db=db)
		print('>%s' % id, file=fo)
		print(seq + '\n', file=fo)
		f.close()
		fo.close()

def make_fasta_from_id(accession, fasta_output_file="./%accesion.fasta", db='/home/kimdh4962/dhpylib/db/uniprot_sprot_0221.fasta'):
	fo = open("%s.fasta" %accession, 'w')
	print(">%s\n%s" %(accession, retrieve_fasta(accession, db=db)), file=fo)
	fo.close()

def AA_321(AA_3):
	AA_321_dic = {"ARG": "R", "HIS": "H", "LYS": "K", "ASP": "D", "GLU": "E", "SER": "S", "THR": "T", "ASN": "N", "GLN": "Q", "CYS": "C", "SEC": "U", "GLY": "G", "PRO": "P", "ALA": "A", "ILE": "I", "LEU": "L", "MET": "M", "PHE": "F", "TRP": "W", "TYR": "Y", "VAL": "V"}
	return AA_321_dic[AA_3.upper()]

# Split FASTA File
def split_fasta(fasta_file, split_fas_dir):
	for seq in SeqIO.parse(fasta_file, "fasta"):
		seq_fname = seq.id.split('|')[-1]
		SeqIO.write(seq, os.path.join(split_fas_dir, seq_fname + '.fasta'), "fasta")

def join_fasta(fasta_source_dir, fasta_file):
	output_handle = open(fasta_file, 'w')
	for fas_file in os.listdir(fasta_source_dir):
		record = SeqIO.read(os.path.join(fasta_source_dir, fas_file), "fasta")
		SeqIO.write(record, output_handle, "fasta")
	output_handle.close()
