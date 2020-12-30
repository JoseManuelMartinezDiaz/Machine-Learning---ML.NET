using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;
using static Microsoft.ML.DataOperationsCatalog;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms.Text;
using PruebaML.NET.Casos.Taxi;

namespace PruebaML.NET
{
  class Program
  {

    static readonly string _dataPath = Path.Combine(Environment.CurrentDirectory, "Datos", "yelp_labelled.txt");


    static void Main(string[] args)
    {
      MLContext mlContext = new MLContext();
      //codigo de ruta, almacen de carga, almacen de descarga y cliente

      RecomendacionPeliculas recomendacionPeliculas = new RecomendacionPeliculas();
      recomendacionPeliculas.iniciar(mlContext);

      //ValoracionRestaurante valoracionRestaurante = new ValoracionRestaurante();
      //valoracionRestaurante.iniciar(mlContext);


      //TaxiFarePrediction taxiFarePrediction = new TaxiFarePrediction();
      //taxiFarePrediction.iniciar(mlContext);

      Console.WriteLine($"Se añade esta linea de codigo para comprobar el funcionamiento de github en visual studio");


    }

  }
}
