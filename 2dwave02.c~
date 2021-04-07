#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <png.h>

#ifdef NDEBUG
#error "Don't turn NDEBUG!!"
#endif

#define frame 10
#define FPS 25.0

const int width = 1000, height = 700;

static double
init2d (int x, int y)
{
  return pow (M_E, -6.0 * (pow ((double) x - (double) width / 2.0,
                                2.0) + pow ((double) y -
                                            (double) height / 2.0,
                                            2.0)) / 10900.0);
}

static void
write_heatmap (double p, png_bytep px)
{
  px[0] = (png_byte) ((p + 1.0) * 255.0 / 2.0);
  px[1] = px[0];
  px[2] = px[0];
  px[3] = 255;
}

int
main ()
{
  double **v, **u;
  const double delta_x = 0.01, c = 89.0, h = 4.0;

  v = malloc (sizeof (double *) * height);
  u = malloc (sizeof (double *) * height);
  assert (v != NULL);

#pragma omp parallel for
  for (int y = 0; y < height; y++)
    {
      v[y] = malloc (sizeof (double) * width);
      u[y] = malloc (sizeof (double) * width);
      assert (u[y] != NULL && v[y] != NULL);
    }

#pragma omp parallel for
  for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
      {
        u[y][x] = init2d (x, y);
        v[y][x] = 0.0;
      }

  png_bytep *png_frame = malloc (sizeof (png_byte) * height);
  assert (png_frame != NULL);

#pragma omp parallel for
  for (int y = 0; y < height; y++)
    assert ((png_frame[y] = malloc (4 * width)) != NULL);

/* This is where the simulation begins */
#pragma omp parallel for ordered schedule(static)
  for (int file = 0; file < frame; file++)
    {
      char buff[50] = { 0 };

      sprintf (buff, "tt-%03d.png", file);

#pragma omp ordered
      {
        printf ("\rWriting png frame: %s %d/%d... ", buff, file, frame - 1);
        fflush (stdout);
        for (int i = 0; i < (int) (1.0 / (FPS * delta_x)); i++)
          {
            for (int x = 0; x < width; x++)
              for (int y = 0; y < height; y++)
                v[y][x] +=
                  delta_x * c * c * (u[((y - 1) < 0) ? 0 : y - 1][x] +
                                     u[((y + 1) ==
                                        height) ? height - 1 : y + 1][x] +
                                     u[y][((x - 1) <
                                           0) ? 0 : x - 1] + u[y][((x + 1) ==
                                                                   width) ?
                                                                  width -
                                                                  1 : x + 1] -
                                     (4.0 * u[y][x])) / (h * h);
#pragma omp parallel for collapse(2)
            for (int x = 0; x < width; x++)
              for (int y = 0; y < height; y++)
                u[y][x] += delta_x * v[y][x];

          }
#pragma omp parallel for
        for (int x = 0; x < width; x++)
          for (int y = 0; y < height; y++)
            write_heatmap (u[y][x], &png_frame[y][x * 4]);
        FILE *fpout = fopen (buff, "wb");
        if (fpout == NULL)
          {
            perror (buff);
            exit (1);;

          }
        png_structp png =
          png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
        assert (png != NULL);

        png_infop info = png_create_info_struct (png);
        assert (info != NULL);

        if (setjmp (png_jmpbuf (png)))
          abort ();
        png_init_io (png, fpout);

// INDENT-OFF
        png_set_IHDR (png,
                      info,
                      width, height,
                      8,
                      PNG_COLOR_TYPE_RGBA,
                      PNG_INTERLACE_NONE,
                      PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
// INDENT-ON
        png_write_info (png, info);

        png_write_image (png, png_frame);
        png_write_end (png, NULL);
        fclose (fpout);
        png_destroy_write_struct (&png, &info);
      }
    }

  free (png_frame);
/* free */
#pragma omp parallel for
  for (int y = 0; y < height; y++)
    {
      free (u[y]);
      free (v[y]);
    }

  free (u);
  free (v);
  puts ("done");
  fflush (stdout);
  return 0;

}
