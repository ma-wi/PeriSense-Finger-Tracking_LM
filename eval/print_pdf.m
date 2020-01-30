function print_pdf(filename)

fig = gcf;
fig.PaperPositionMode = 'auto';
fig_pos = fig.PaperPosition;
fig.PaperSize = [fig_pos(3) fig_pos(4)];
fig.Renderer='Painters';
img = getframe(gcf);
imwrite(img.cdata, [filename, '.png']);
print(fig, sprintf('%s.pdf', filename), '-dpdf', '-r0');

end