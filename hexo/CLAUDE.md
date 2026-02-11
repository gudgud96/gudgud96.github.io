When asked to make changes in this branch and later push to Github Pages (gudgud96.github.io), follow these steps:

1. Ensure changes are made on the markdown .md files in `develop` branch
2. Run `hexo generate`, you should see a newly made `public` folder with the actual HTML files
3. Move the `public` folder to two layers above (../..), replace the `../../blog_sandbox` folder with this public folder
4. Commit the changes made in the develop branch, push to GitHub
5. Go to ../ which is the root of gudgud96.github.io/, and change to master branch
6. From outside the root, copy everything in ../blog_sandbox/* into the root directory gudgud96.github.io/. Please check if this is done properly, if so, you should see gudgud96.github.io/index.html. Verify this step.
7. Commit all changes, and push to Github.
