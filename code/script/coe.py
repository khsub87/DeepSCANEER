import os, sys
from script import lib_path
import subprocess

def check_conf_file():
	f = open("./script/coe/Energetics.properties")
	line_list = f.readlines()
	HOME_DIR = line_list[4].strip().split("=")[1]
	f.close()
	if not os.path.isdir(HOME_DIR):
		new_HOME_DIR = os.path.abspath("./script/coe")
		fo = open("./script/coe/Energetics.properties", 'w')
		for i, line in enumerate(line_list):
			if i == 4: print ("HOME_DIRECTORY=%s" %new_HOME_DIR, file=fo)
			else: print (line.strip(), file=fo)
		fo.close()

check_conf_file()

class ProcCoe:
    # input aln, out coe file path are required
    def __init__(self, aln_file, coe_file, algorithm="McBASC"):
        self.algorithm = algorithm
        self.input_aln_file = os.path.abspath(aln_file)
        self.output_coe_file = os.path.abspath(coe_file)
        self.result = []

    # Available Algorithm: McBASC, ELSC, SCA, MI, OMES
    def run(self):
        if sys.platform == 'win32': # Windows Environment
            jexePath = 'java.exe'
            cmd_apd = ' & '
        else: # Linux Environment
            jexePath = "java"
            cmd_apd = ';'
        jClassPath = os.path.join("./script", "coe")
        jClassAlgorithm = "covariance.algorithms.%s" % self.algorithm
        exejClass =[jexePath, jClassAlgorithm, self.input_aln_file, self.output_coe_file]
        result=subprocess.run(exejClass, cwd=jClassPath, capture_output=True, text=True, check=True)
		
    # Parse coe-calculation result
    def parse(self):
        f = open(self.output_coe_file, 'r')
        f.readline()
        for line in f.readlines():
            fields = line.split()
            self.result.append((int(fields[0]), int(fields[1]), float(fields[2])))
        f.close()
        return self.result

    def convertResPos(self, res_pos_dic):
        f = open(self.output_coe_file, 'w')
        print ("residue_1\tresidue_2\tscore", file=f)
        for cell in self.result:
            print ('%d\t%d\t%s' %\
            (res_pos_dic[cell[0]]+1, res_pos_dic[cell[1]]+1, str(cell[2])), file=f)
        f.close()
			
