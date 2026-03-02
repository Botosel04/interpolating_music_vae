{-# LANGUAGE DeriveAnyClass #-}
{-# LANGUAGE DeriveGeneric #-}
{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE MultiParamTypeClasses #-}
{-# LANGUAGE RecordWildCards #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

import Control.Monad ( forM_, forM, when, (<=<) )
import Control.Monad.Cont ( ContT (..) )
import Data.List (foldl')
import System.Environment ( getArgs )
import GHC.Generics
import Pipes hiding ( (~>) )
import qualified Pipes.Prelude as P
import Text.Printf ( printf )
import Torch
import Torch.Serialize
import Torch.Typed.Vision ( initMnist, MnistData )
import qualified Torch.Vision as V
import Torch.Lens ( HasTypes (..)
                  , over
                  , types )
import Prelude hiding ( exp )
import qualified Torch.Optim.CppOptim as Cpp
import Data.Default.Class
import Control.Exception
import System.CPUTime
import System.Directory (doesFileExist)
data VAESpec = VAESpec
  {
    fc1 :: LinearSpec,
    fcMu :: LinearSpec,
    fcSigma :: LinearSpec,
    fc5 :: LinearSpec,
    fc6 :: LinearSpec
  }
  deriving (Show, Eq)

latent_size = 2

myConfig =
  VAESpec
    (LinearSpec 784 400)
    (LinearSpec 400 latent_size)
    (LinearSpec 400 latent_size)
    (LinearSpec latent_size 400)
    (LinearSpec 400 784)

data VAE = VAE
  { l1 :: Linear,
    lMu :: Linear,
    lSigma :: Linear,
    l5 :: Linear,
    l6 :: Linear
  }
  deriving (Generic, Show, Parameterized)

instance Randomizable VAESpec VAE where
  sample VAESpec {..} =
    VAE
      <$> sample fc1
      <*> sample fcMu
      <*> sample fcSigma
      <*> sample fc5
      <*> sample fc6

decode :: VAE -> Tensor -> Tensor
decode VAE {..} =
  linear l5
  ~> relu
  ~> linear l6
  ~> sigmoid

encode :: VAE -> Tensor -> (Tensor, Tensor)
encode VAE {..} x0 =
  let enc_ =
        linear l1
         ~> relu
      x1 = enc_ x0
      mu = linear lMu x1
      logSigma = linear lSigma x1
  in (mu, logSigma)

vaeForward :: VAE -> Bool -> Tensor -> IO (Tensor, Tensor, Tensor)
vaeForward net@(VAE {..}) _ x0 = do
  let (mu, logSigma) = encode net x0
      sigma = exp (0.5 * logSigma)
  eps <- toLocalModel' <$> randnLikeIO sigma
  let z = (eps `mul` sigma) `add` mu
      reconstruction = decode net z
  return (reconstruction, mu, logSigma)

(~>) :: (a -> b) -> (b -> c) -> a -> c
f ~> g = g. f

toLocalModel :: forall a. HasTypes a Tensor => Device -> DType -> a -> a
toLocalModel device' dtype' = over (types @Tensor @a) (toDevice device')

fromLocalModel :: forall a. HasTypes a Tensor => a -> a
fromLocalModel = over (types @Tensor @a) (toDevice (Device CPU 0))

toLocalModel' :: forall a. HasTypes a Tensor => a -> a
toLocalModel' = toLocalModel (Device CPU 0) Float


bceLossWeighted :: Float -> Tensor -> Tensor -> Tensor
bceLossWeighted weight target x =
  let 
      posLoss = (asTensor weight) * target * Torch.log(1e-10 + x)
      negLoss = (1 - target) * Torch.log(1e-10 + 1 - x)
      r = -sumAll (posLoss + negLoss)
  in r


vaeLoss :: Float -> Tensor -> Tensor -> Tensor -> Tensor -> Tensor
vaeLoss beta recon_x x mu logSigma = reconLoss + asTensor beta * kld
  where
    reconLoss = bceLossWeighted 1.0 x recon_x 
    kld = -0.5 * sumAll (1 + logSigma - pow (2 :: Int) mu - exp logSigma)

trainLoop
  :: Optimizer o
  => Float -> (VAE, o) -> LearningRate -> ListT IO (Tensor, Tensor) -> IO (VAE, o)
trainLoop beta (model0, opt0) lr = P.foldM step begin done. enumerateData
  where
    step :: Optimizer o => (VAE, o) -> ((Tensor, Tensor), Int) -> IO (VAE, o)
    step (model, opt) args = do
      let ((x, _), iter) = toLocalModel' args
          x' = x / 255.0
      (recon_x, mu, logSigma) <- vaeForward model False x'
      let loss = vaeLoss beta recon_x x' mu logSigma
      when (iter `mod` 100 == 0) $ do
        putStrLn
          $ printf "Batch: %d | Loss: %.2f" iter (asValue loss :: Float)
      runStep model opt loss lr
    done = pure
    begin = pure (model0, opt0)

train :: Float -> V.MNIST IO -> Int -> VAE -> IO VAE
train beta trainMnist epochs net0 = do
    optimizer <- Cpp.initOptimizer adamOpt net0
    (net', _) <- foldLoop (net0, optimizer) epochs $ \(net', optState) _ ->
      runContT (streamFromMap dsetOpt trainMnist)
      $ trainLoop beta (net', optState) 0.0 . fst
    return net'
  where
    dsetOpt = datasetOpts workers
    workers = 2
    adamOpt =
        def
          { Cpp.adamLr = learningRate,
            Cpp.adamBetas = (0.9, 0.999),
            Cpp.adamEps = 1e-8,
            Cpp.adamWeightDecay = 0,
            Cpp.adamAmsgrad = False
          } ::
          Cpp.AdamOptions

save' :: VAE -> FilePath -> IO ()
save' net = save (map toDependent. flattenParameters $ net)

load' :: FilePath -> IO VAE
load' fpath = do
  params <- mapM makeIndependent <=< load $ fpath
  net0 <- sample myConfig
  return $ replaceParameters net0 params

testLatentSpace :: FilePath -> V.MNIST IO -> VAE -> IO ()
testLatentSpace fn testStream net = do
      runContT (streamFromMap (datasetOpts 2) testStream) $ recordPoints fn net. fst

recordPoints :: FilePath -> VAE -> ListT IO (Tensor, Tensor) -> IO ()
recordPoints logname net = P.foldM step begin done. enumerateData
  where
    step :: () -> ((Tensor, Tensor), Int) -> IO ()
    step () args = do
      let ((input, labels), i) = toLocalModel' args
          (encMu, _) = encode net input
      let s = toStr $ Torch.cat (Dim 1) [reshape [-1, 1] labels, encMu]
      appendFile logname s
      return ()
    done () = pure ()
    begin = pure ()

toStr :: Tensor -> String
toStr dec =
    let a = asValue dec :: [[Float]]
        b = map (unwords. map show) a
     in unlines b

time :: IO t -> IO t
time a = do
    start <- getCPUTime
    v <- a
    end   <- getCPUTime
    let diff = fromIntegral (end - start) / (10^12)
    printf "Computation time: %0.3f sec\n" (diff :: Double)
    return v

learningRate :: Double
learningRate = 1e-4

loadSong :: FilePath -> IO Tensor
loadSong path = do
    content <- readFile path
    let vals = map read (lines content) :: [Float]
    return $ toLocalModel' $ reshape [1, 784] $ asTensor vals

interpolateSequence :: VAE -> FilePath -> FilePath -> IO ()
interpolateSequence net file1 file2 = do
    putStrLn $ "Morphing from " ++ file1 ++ " to " ++ file2 ++ "..."
    t1 <- loadSong file1
    t2 <- loadSong file2
    let (mu1, _) = encode net t1
        (mu2, _) = encode net t2
        steps = 20 :: Int
    let zs = map (\i -> 
                    let alpha = (fromIntegral i) / (fromIntegral steps) :: Float
                        term1 = mu1 `mul` (asTensor (1.0 - alpha))
                        term2 = mu2 `mul` (asTensor alpha)
                    in term1 `add` term2
                 ) [0..steps]
    let decodedList = map (decode net) zs
        finalTensor = Torch.cat (Dim 0) decodedList
        outName = "morph_sequence.txt"
    writeFile outName (toStr finalTensor)
    putStrLn $ "Done! Saved entire sequence to: " ++ outName

main = do
    (trainData, testData) <- initMnist "music/data_synthetic"

    beta_: _ <- getArgs
    putStrLn $ "beta = " ++ beta_
    let beta = read beta_
        
        trainMnistStream = V.MNIST { batchSize = 128, mnistData = trainData }
        testMnistStream = V.MNIST { batchSize = 99, mnistData = testData }
        epochs = 100
        modelFile = printf "VAE-music-beta_%s.ht" beta_
        logname = printf "beta_%s.log" beta_ :: String

    
    let trainMode = True 

    net <- if trainMode
        then do
            putStrLn "--- STARTING TRAINING ---"
            net0 <- toLocalModel' <$> sample myConfig
            trainedNet <- time $ train beta trainMnistStream epochs net0
            save' trainedNet modelFile
            putStrLn $ "Model saved to " ++ modelFile
            return trainedNet
        else do
            putStrLn $ "--- LOADING MODEL: " ++ modelFile ++ " ---"
            load' modelFile
    testLatentSpace logname testMnistStream net

    putStrLn "Generating latent space reconstruction..."
    let xs = [-3, -2.7 .. 3 :: Float]
        zs = [[x, y] | x <- xs, y <- xs]
        decoded = Torch.cat (Dim 0) $
                    map (decode net . toLocalModel' . asTensor . (:[])) zs
    writeFile (printf "latent_reconstruction_beta_%s.txt" beta_) (toStr decoded)

    -- Interpolation
    interpolateSequence net "music/synthetic_songs/song_2.txt" "music/synthetic_songs/song_0.txt"
    
    putStrLn "Done"
