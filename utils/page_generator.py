# Generate project and thought article pages
# Author: gudgud96

import os
import sys

PROJECT_PATH = "../projects/"
THOUGHTS_PATH = "../thoughts/"

def main(argv, path):
  os.system("python -m markdown " + path + argv[0] + " > " + path + "temp.html")
  with open(path + 'project_template.html', 'r+') as templatefile:
    template = templatefile.read()
  with open(path + "temp.html", "r+") as contentfile:
    content = contentfile.read()
  with open(path + argv[1], "w+") as output:
    output.write(template.replace("{content}", content).replace("img", 'img class="in-text"'))
    # add "in-text" class to all images in content
  os.remove(path + "temp.html")
  print("Successfully generated.")


if __name__ == "__main__":
  print("Usage: python page_generator.py -<projects/thoughts> <.md file name> <.html file name>")
  if sys.argv[1][1:] == "projects":
    main(sys.argv[2:], PROJECT_PATH)
  else:
    main(sys.argv[2:], THOUGHTS_PATH)
