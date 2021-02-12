scp scuri@multimodularity.ethz.ch:/local/scuri/rhucrl/applications/*.json .
scp scuri@lazy.ethz.ch:/local/scuri/rhucrl/applications/*.json .
scp scuri@shallow.ethz.ch:/local/scuri/rhucrl/applications/*.json .
scp scuri@login.leonhard.ethz.ch:/cluster/project/infk/krause/scuri/rhucrl/applications/*.json .

python merge.py
