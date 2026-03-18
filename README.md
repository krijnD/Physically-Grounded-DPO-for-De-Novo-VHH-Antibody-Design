# Physically-Grounded-DPO-for-De-Novo-VHH-Antibody-Design

Downloading ANDD:
wget -O ANDD_pdb.zip "https://zenodo.org/records/18151718/files/ANDD_pdb.zip?download=1"

Sabdab nano summary:
wget https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab/summary/nanobody/ -O sabdab_nano_summary.tsv

Install Rosseta: 

python -m venv /projects/0/hpmlprjs/interns/krijn/venvs/rosetta

pip install pyrosetta \
  --find-links https://graylab.jhu.edu/download/PyRosetta4/archive/release-quarterly/release