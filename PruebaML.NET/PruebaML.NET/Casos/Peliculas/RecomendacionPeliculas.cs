using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Trainers;

namespace PruebaML.NET
{
  class RecomendacionPeliculas
  {

    public RecomendacionPeliculas()
    {

    }

    public void iniciar(MLContext mlContext)
    {
      (IDataView trainingDataView, IDataView testDataView) = LoadData(mlContext);
      ITransformer model = BuildAndTrainModel(mlContext, trainingDataView);
      
      EvaluateModel(mlContext, testDataView, model);
      UseModelForSinglePrediction(mlContext, model);

      SaveModel(mlContext, trainingDataView.Schema, model);
    }

    private static (IDataView training, IDataView test) LoadData(MLContext mlContext)
    {
      //var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "DETALLE_RUTAS_train.csv");
      //var testDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "DETALLE_RUTAS_tets.csv");

      var trainingDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "recommendation-ratings-train.csv");
      var testDataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "recommendation-ratings-test.csv");

      IDataView trainingDataView = mlContext.Data.LoadFromTextFile<MovieRating>(trainingDataPath, hasHeader: true, separatorChar: ',');
      IDataView testDataView = mlContext.Data.LoadFromTextFile<MovieRating>(testDataPath, hasHeader: true, separatorChar: ',');

      return (trainingDataView, testDataView);
    }

    private static ITransformer BuildAndTrainModel(MLContext mlContext, IDataView trainingDataView)
    {
      IEstimator<ITransformer> estimator = mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "IDEncoded", inputColumnName: "ID")
                                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "id_almacenEncoded", inputColumnName: "id_almacen"));

      var options = new MatrixFactorizationTrainer.Options
      {
        MatrixColumnIndexColumnName = "IDEncoded",
        MatrixRowIndexColumnName = "id_almacenEncoded",
        LabelColumnName = "Label",
        NumberOfIterations = 30,
        ApproximationRank = 100
      };

      var trainerEstimator = estimator.Append(mlContext.Recommendation().Trainers.MatrixFactorization(options));

      Console.WriteLine("=============== Training the model ===============");
      ITransformer model = trainerEstimator.Fit(trainingDataView);

      return model;
    }

    public static void EvaluateModel(MLContext mlContext, IDataView testDataView, ITransformer model)
    {
      Console.WriteLine("=============== Evaluating the model ===============");
      var prediction = model.Transform(testDataView);

      var metrics = mlContext.Regression.Evaluate(prediction, labelColumnName: "Label", scoreColumnName: "Score");

      Console.WriteLine("Root Mean Squared Error : " + metrics.RootMeanSquaredError.ToString());
      Console.WriteLine("RSquared: " + metrics.RSquared.ToString());
    }

    public static void UseModelForSinglePrediction(MLContext mlContext, ITransformer model)
    {
      Console.WriteLine("=============== Making a prediction ===============");
      var predictionEngine = mlContext.Model.CreatePredictionEngine<MovieRating, MovieRatingPrediction>(model);

      var testInput = new MovieRating { ID = 52, id_almacen = 84 };

      var movieRatingPrediction = predictionEngine.Predict(testInput);
      
      //Console.WriteLine("Movie " + testInput.ID + " is recommended for user " + testInput.id_almacen);
      if (Math.Round(movieRatingPrediction.Score, 1) > 3.5)
      {
        Console.WriteLine("ID " + testInput.ID + " is recommended for almacen " + testInput.id_almacen);
      }
      else
      {
        Console.WriteLine("ID " + testInput.ID + " is not recommended for almacen " + testInput.id_almacen);
      }
    }

    public static void SaveModel(MLContext mlContext, DataViewSchema trainingDataViewSchema, ITransformer model)
    {
      var modelPath = Path.Combine(Environment.CurrentDirectory, "Datos", "MovieRecommenderModel.zip");

      Console.WriteLine("=============== Saving the model to a file ===============");
      mlContext.Model.Save(model, trainingDataViewSchema, modelPath);


    }
  }
}
