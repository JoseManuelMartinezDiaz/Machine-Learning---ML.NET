using Microsoft.ML.Data;

namespace PruebaML.NET.Casos.Taxi
{
  class TaxiTrip
  {
    [LoadColumn(0)]
    public string VendorId;

    [LoadColumn(1)]
    public string RateCode;

    [LoadColumn(2)]
    public float PassengerCount;

    [LoadColumn(3)]
    public float TripTime;

    [LoadColumn(4)]
    public float TripDistance;

    [LoadColumn(5)]
    public string PaymentType;

    [LoadColumn(6)]
    public float FareAmount;
  }

  class TaxiTripFarePrediction
  {
    [ColumnName("Score")]
    public float FareAmount;
  }

  internal class SingleTaxiTripSample
  {
    internal static readonly TaxiTrip Trip1 = new TaxiTrip
    {
      VendorId = "VTS",
      RateCode = "1",
      PassengerCount = 1,
      TripDistance = 10.33f,
      PaymentType = "CSH",
      FareAmount = 0 // predict it. actual = 29.5
    };
  }
}
