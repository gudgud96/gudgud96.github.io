import os
import sys

PATH = "../projects/"

def main(argv):
  os.system("python -m markdown " + PATH + argv[0] + " > " + PATH + "temp.html")
  with open(PATH + 'project_template.html', 'r+') as templatefile:
    template = templatefile.read()
  with open(PATH + "temp.html", "r+") as contentfile:
    content = contentfile.read()
  with open(PATH + argv[1], "w+") as output:
    output.write(template.replace("{content}", content).replace("img", 'img class="in-text"'))
    # add "in-text" class to all images in content


if __name__ == "__main__":
  main(sys.argv[1:])