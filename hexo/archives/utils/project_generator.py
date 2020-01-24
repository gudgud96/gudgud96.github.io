# Generate project page as table of contents 
# Author: gudgud96

import os
import sys

TEMPLATE_PATH = "../templates/"
MAIN_PATH = "../"
PROJECT_NUM = "{project-num}"
PROJECT_TITLE = "{project-title}"
STATUS = "{status}"
STATUS_PERCENT = "{status-percent}"
PROJECT_DESCRIPTION = "{project-description}"
PROJECTS = "{projects}"
PROJECT_FILENAME = "{project-filename}"
CONSTANTS_LIST = [PROJECT_NUM, PROJECT_TITLE, STATUS, STATUS_PERCENT, PROJECT_DESCRIPTION, PROJECT_FILENAME]

def main():
  with open(TEMPLATE_PATH + 'projects_markup.md', 'r+') as markupfile:
    markup = markupfile.readlines()
  with open(TEMPLATE_PATH + 'projects_template.html', 'r+') as templatefile:
    template = templatefile.read()
  with open(TEMPLATE_PATH + 'project_block_template.html', 'r+') as blocktemplatefile:
    blocktemplate = blocktemplatefile.read()
  
  with open(MAIN_PATH + 'projects.html', "w+") as output:
    output_content = ""
    block = ""
    for i in range(len(markup)):
      if i % 6 == 0:
        output_content += block
        block = blocktemplate
      block = block.replace(CONSTANTS_LIST[i % 6], markup[i].replace('\n',''))
    output_content += block

    output.write(template.replace(PROJECTS, output_content).replace("img", 'img class="in-text"'))
    # add "in-text" class to all images in content

  print("Successfully generated.")


if __name__ == "__main__":
  print("Usage: python project_generator.py")
  main()
