namespace PruebaML.NET.Casos.Taxi
{
  internal class GetDataFromCsv
  {
    private string testDataSetPath;
    private int totalNumber;

    public GetDataFromCsv(string testDataSetPath, int totalNumber)
    {
      this.testDataSetPath = testDataSetPath;
      this.totalNumber = totalNumber;
    }
  }
}