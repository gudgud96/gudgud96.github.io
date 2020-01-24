# Generate project page as table of contents 
# Author: gudgud96

import os
import sys

TEMPLATE_PATH = "../templates/"
MAIN_PATH = "../"
TITLE_LINE = '<li><i class="fh5co-tab-menu-icon"></i>{thought-title}</li>'
TITLE_TAB = "{title-tab}"
THOUGHTS = "{thoughts}"
THOUGHT_TITLE = "{thought-title}"
THOUGHT_DESCRIPTION = "{thought-description}"
THOUGHT_FILENAME = "{thought-filename}"
CONSTANTS_LIST = [THOUGHT_TITLE, THOUGHT_DESCRIPTION, THOUGHT_FILENAME]

def main():
  with open(TEMPLATE_PATH + 'thoughts_markup.md', 'r+') as markupfile:
    markup = markupfile.readlines()
  with open(TEMPLATE_PATH + 'thoughts_template.html', 'r+') as templatefile:
    template = templatefile.read()
  with open(TEMPLATE_PATH + 'thoughts_block_template.html', 'r+') as blocktemplatefile:
    blocktemplate = blocktemplatefile.read()
  
  with open(MAIN_PATH + 'thoughts.html', "w+") as output:
    output_content = ""
    title_tab = ""
    block = ""
    for i in range(len(markup)):
      if i % 3 == 0:
        output_content += block
        block = blocktemplate
        title_tab += TITLE_LINE.replace(THOUGHT_TITLE, markup[i]) + '\n'
      block = block.replace(CONSTANTS_LIST[i % 3], markup[i].replace('\n',''))
    output_content += block

    output.write(template.replace(THOUGHTS, output_content).replace(TITLE_TAB, title_tab).replace("img", 'img class="in-text"'))
    # add "in-text" class to all images in content

  print("Successfully generated.")


if __name__ == "__main__":
  print("Usage: python thoughts_generator.py")
  main()
