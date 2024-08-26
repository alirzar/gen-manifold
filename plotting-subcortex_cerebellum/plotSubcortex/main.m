%% main
mkdir("figures/behavior/")
mkdir("figures/generalization/")
mkdir("figures/measures/")
mkdir("figures/reference/")
mkdir("figures/seed/")
mkdir("figures/seed-network/")

run("ref_grad_parcellationPlot.m")
run("RLGenPLotMeasures.m")
run("RLGenPLot.m")
run("RLGenPlotSeed.m")
run("RLGenPLotSeedNetwork.m")