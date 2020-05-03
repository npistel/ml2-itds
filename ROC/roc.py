import ROOT

ROOT.TMVA.Tools.Instance()

inputFile = ROOT.TFile.Open("inputdata.root")
outputFile = ROOT.TFile.Open("TMVAOutputCV.root", "RECREATE")

factory = ROOT.TMVA.Factory("TMVAClassification", outputFile,
                            "!V:ROC:!Correlations:!Silent:Color:!DrawProgressBar:AnalysisType=Classification")

loader = ROOT.TMVA.DataLoader("dataset")
loader.AddVariable("var1")
loader.AddVariable("var2")
loader.AddVariable("var3")

tsignal = inputFile.Get("Sig")
tbackground = inputFile.Get("Bkg")

loader.AddSignalTree(tsignal, 1.0)
loader.AddBackgroundTree(tbackground, 1.0)
loader.PrepareTrainingAndTestTree(ROOT.TCut(""), ROOT.TCut(""),
                                   "nTrain_Signal=1000:nTrain_Background=1000:SplitMode=Random:NormMode=NumEvents:!V")

# Boosted Decision Trees
factory.BookMethod(loader, ROOT.TMVA.Types.kBDT, "BDT",
                   "!V:NTrees=200:MinNodeSize=2.5%:MaxDepth=2:BoostType=AdaBoost:AdaBoostBeta=0.5:UseBaggedBoost:BaggedSampleFraction=0.5:SeparationType=GiniIndex:nCuts=20")

# Multi-Layer Perceptron (Neural Network)
factory.BookMethod(loader, ROOT.TMVA.Types.kMLP, "MLP",
                   "!H:!V:NeuronType=tanh:VarTransform=N:NCycles=100:HiddenLayers=N+5:TestRate=5:!UseRegulator")

factory.TrainAllMethods()

factory.TestAllMethods()
factory.EvaluateAllMethods()

c = factory.GetROCCurve(loader)
c.Draw()

raw_input()
