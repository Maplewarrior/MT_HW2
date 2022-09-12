import optparse
import sys
from collections import defaultdict

optparser = optparse.OptionParser()
optparser.add_option("-d", "--data", dest="train", default="data/hansards", help="Data filename prefix (default=data)")
optparser.add_option("-e", "--english", dest="english", default="e", help="Suffix of English filename (default=e)")
optparser.add_option("-f", "--french", dest="french", default="f", help="Suffix of French filename (default=f)")
optparser.add_option("-t", "--threshold", dest="threshold", default=0.5, type="float", help="Threshold for aligning with Dice's coefficient (default=0.5)")
optparser.add_option("-n", "--num_sentences", dest="num_sents", default=1000, type="int", help="Number of sentences to use for training and alignment")
optparser.add_option("-i", "--iterations", dest="n_iter", default = 5, type="int", help="Number of iterations to run algorithm")
(opts, _) = optparser.parse_args()
f_data = "%s.%s" % (opts.train, opts.french)
e_data = "%s.%s" % (opts.train, opts.english)



sys.stderr.write("Training EM Algorithm...")
bitext = [[sentence.strip().split() for sentence in pair] for pair in zip(open(f_data), open(e_data))][:opts.num_sents]
f_count = defaultdict(int)
e_count = defaultdict(int)
fe_count = defaultdict(int)

# Get counts
for (n, (f, e)) in enumerate(bitext):
  for f_i in set(f):
    f_count[f_i] += 1
    for e_j in set(e):
      fe_count[(f_i,e_j)] += 1
  for e_j in set(e):
    e_count[e_j] += 1
  if n % 500 == 0:
    sys.stderr.write(".")

########### EM ALGORITHM ###########
# Initialize dictionary of estimated probabilites
theta_0 = 1/len(e_count.keys())
thetas = defaultdict(float)
thetas = {(f_i, e_j): theta_0 for (f_i, e_j) in fe_count.keys()}


for k in range(opts.n_iter):
    for (f, e) in bitext: # Loop over all sentences
        for (i, f_i) in enumerate(f): # loop over all french words
            Z = 0
            # print("restarting")
            for (j, e_j) in enumerate(e): # loop over all english words
                Z += thetas[(f_i, e_j)]
            for (j, e_j) in enumerate(e):
                c = thetas[(f_i, e_j)]/Z
                fe_count[(f_i, e_j)] += c
                e_count[e_j]+=c
                
    for (f_i, e_j) in thetas.keys():
        thetas[(f_i, e_j)] = fe_count[(f_i, e_j)]/e_count[e_j]

        
############### GET RESULT ################
alignments_dict = defaultdict(int)

for iter, (f, e) in enumerate(bitext):
    # print("iternation n.o:", iter)
    # print(f)
    for (i, f_i) in enumerate(f):
        best_prob = 0
        best_align = [0,0]
        e_star = None
        for (j, e_j) in enumerate(e):
            
            # print("FRENCH:\n", f_i)
            # print("ENGLISH:\n", e_j)
            if thetas[(f_i, e_j)] > best_prob:
                best_prob = thetas[(f_i, e_j)]
                best_align = [i, j]
                e_star = e_j
                
        # best_align = tuple(best_align)
        
        # print("French & english word:\n", f_i, "&", e_star)
        # print("prob:", best_prob)
        # print("alignment:", tuple(best_align))
        sys.stdout.write(str(best_align[0]) + "-" + str(best_align[1]) + " ")
        alignments_dict[(f_i,e_star)] = tuple(best_align)
    sys.stdout.write("\n")




