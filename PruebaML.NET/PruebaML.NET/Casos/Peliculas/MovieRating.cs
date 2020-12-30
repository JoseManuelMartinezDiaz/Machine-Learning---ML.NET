using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace PruebaML.NET
{
  class MovieRating
  {

    [LoadColumn(0)]
    public float ID;
    [LoadColumn(1)]
    public float id_almacen;
    [LoadColumn(2)]
    public float Label;
  }

  public class MovieRatingPrediction
  {
    public float Score;
    public float Label;
  }


}
