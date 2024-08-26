from genman import (connectivity, gradients, centering_umap)
from genman.analyses import (reference, measures, eccentricity, generalization, 
                             seed, behaviour, seed_network)
import shutil

connectivity.main()
gradients.main()
centering_umap.main()

# reference analyses
reference.main()
measures.main()

# subject-level analyses
eccentricity.main()
generalization.main()
seed.main()

behaviour.main()
seed_network.main()

shutil.copytree('../results/', '../plotting-subcortex_cerebellum/plotCerebellum/results/', dirs_exist_ok=True)
shutil.copytree('../results/', '../plotting-subcortex_cerebellum/plotSubcortex/results/', dirs_exist_ok=True)
