class FileManager():

  def openFile(path):
    with open(path) as dataset:
      df = pd.read_excel(path)
      return df

  def printOnFile(line):
    with open("PerceptronTests.txt", "a+", encoding="utf-8") as file:
      file.write(line + "\n")