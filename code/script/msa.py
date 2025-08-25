import os, sys
from Bio import AlignIO
from Bio.Align import MultipleSeqAlignment
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from script.__init__ import *

class ProcMsa:
    def __init__(self, input_fasta, output_aln, output_tree, method="muscle"):
        self.input_fasta = input_fasta
        self.output_aln = output_aln
        self.output_tree = output_tree
        self.method = method
        self.result = None
    
    def run(self):
        def execClustalw(self):
            clustalw_exe = os.path.join(lib_path, get_executable_arch('msa_exec/<arch>/clustalw2'))
            os.system("%s -infile=%s -outfile=%s -newtree=%s -outorder=input 2>err1.txt" %\
                       (clustalw_exe, self.input_fasta, self.output_aln, self.output_tree))
    
        def execMuscle(self):
            muscle_exe = os.path.join(lib_path, get_executable_arch('msa_exec/<arch>/muscle'))
            output_aln_tmp = self.output_aln + '_tmp'
            output_aln_fas = self.output_aln + '_fas'
            os.system("%s -in %s -out %s -tree2 %s" %\
                        (muscle_exe, self.input_fasta, output_aln_tmp, self.output_tree))
	    ### muscle bug fix (stable.py)
            os.system('%s %s %s %s > %s' % (sys.executable.replace('pythonw', 'python'),\
                                         os.path.join(lib_path, 'msa_exec/muscle_stable_patch.py'),\
                                         self.input_fasta,\
                                         output_aln_tmp,\
                                         output_aln_fas))
            AlignIO.convert(output_aln_fas, 'fasta', self.output_aln, 'clustal')
            
            os.unlink(output_aln_tmp)
            os.unlink(output_aln_fas)
        
        def execTCoffee(self):
            tcoffee_exe = os.path.join(lib_path, get_executable_arch('msa_exec/<arch>/t_coffee'))
            os.system("%s -infile=%s -outfile=%s -newtree=%s -outorder=input" %\
                       (tcoffee_exe, self.input_fasta, self.output_aln, self.output_tree)) 
        
        if self.method == "clustalw": execClustalw(self)
        elif self.method == "muscle": execMuscle(self)
        else: execTCoffee(self)
    
    # Parse Alignment File with AlignIO     
    def parse(self):
        self.result = AlignIO.read(self.output_aln, "clustal")
        return self.result
    
    def makeCNinput(self, output_file, query_id, cutoff=0.2):
        out_f = open(output_file, "w")
        
        msa_dic = {}
        for seq in self.result:
            if seq.id == query_id[:30]:
                #msa_dic[query_id] = seq.seq.tostring()
                msa_dic[query_id] = str(seq.seq)
            else:
	            #msa_dic[seq.id] = seq.seq.tostring()
	            msa_dic[seq.id] = str(seq.seq)
        query_seq = msa_dic[query_id]
        #print(msa_dic)
        # MSA position
        res_pos_dic = {}
        msa_pos, query_res_pos = (0, 0)
        column_len = float(len(self.result))
        val_res = []
        for msa_num in range(len(query_seq)):
            column = list(map(lambda x: x[msa_num], msa_dic.values()))
            #print(column)
            #column = self.result.get_column(msa_num)
            if query_seq[msa_num] != '-':
                if column.count('-') / column_len < cutoff:
                    res_pos_dic[msa_pos] = query_res_pos
                    msa_pos += 1
                    val_res.append(msa_num) 
                query_res_pos += 1
        
        for seq_id in msa_dic.keys():
            seq = ''
            for res_num in val_res:
                seq += msa_dic[seq_id][res_num]
            print (seq_id + ' ' + seq, file=out_f)
        # print(self.result)
        # print('column_len is %s'%column_len)
        # print(query_seq)
        # print('len is %s'%len(query_seq.replace('-', '')))

        # print('msa_pos is %s'%msa_pos)
        # print('query_pos is %s'%query_res_pos)
        # print('len is %s'%len(query_seq.replace('-', '')))
        out_f.close()    
        #return res_pos_dic, int(column_len)
        return res_pos_dic, len(query_seq.replace('-', ''))
    
    def makeCSinput(self, output_file, cut_num=None):
        if cut_num != None:
            AlignIO.write(self.result[:cut_num], output_file, 'clustal')
        else:
            AlignIO.write(self.result, output_file, 'clustal')
