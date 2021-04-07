#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <signal.h>
#include <assert.h>
#include <png.h>

#ifdef NDEBUG
#error "Don't turn NDEBUG!!"
#endif

#define frame 10
#define FPS 25.0

char wav_struct[] = {
  'R', 'I', 'F', 'F',
  0, 0, 0, 0,                   /* chunksize */
  'W', 'A', 'V', 'E',
  'f', 'm', 't', ' ',           /* subchunk1id */
  16, 0, 0, 0,                  /* subchunk1size */
  1, 0,                         /* audioformat */
  1, 0,                         /* numchannels */
  68, 172, 0, 0,                /* samplerate 44100 */
  136, 88, 1, 0,                /* byterate 88200 */
  2, 0,                         /* blockalign */
  16, 0,                        /* bitspersample */
  'd', 'a', 't', 'a',           /* subchunk2id */
  0, 0, 0, 0                    /* subchunk2size */
};

double **v = NULL, **u = NULL;
int16_t *pcmin = NULL, *pcmout = NULL;
png_bytep *png_frame = NULL;
const int width = 1000, height = 700;
int pcm_ctr = 0;

void
write_wav (void)
{
  uint32_t length = 32 + (pcm_ctr << 1);
  uint32_t subchunk2size = pcm_ctr << 1;
  FILE *fpout = fopen ("write.wav", "wb");
  fwrite (wav_struct, sizeof (wav_struct), 1, fpout);
  fwrite (pcmout, sizeof (int16_t), pcm_ctr, fpout);
  fseek (fpout, 4, SEEK_SET);
  fwrite (&length, sizeof (uint32_t), 1, fpout);
  fseek (fpout, 40, SEEK_SET);
  fwrite (&subchunk2size, sizeof (uint32_t), 1, fpout);

  fclose (fpout);
}

void
intHandler (int c)
{
  printf ("\nSignal is catched, exiting normally...\n");
  write_wav ();
  free (pcmin);
  free (pcmout);

  if (u != NULL)
    for (int y = 0; y < height; y++)
      free (u[y]);

  if (v != NULL)
    for (int y = 0; y < height; y++)
      free (v[y]);

  if (png_frame != NULL)
    for (int y = 0; y < height; y++)
      free (png_frame[y]);

  free (png_frame);
  free (u);
  free (v);

  exit (1);
}

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
  px[0] = (png_byte) (((p / 32768.0) + 1.0) * 255.0 / 2.0);
  px[1] = px[0];
  px[2] = px[0];
  px[3] = 255;
}

int
main (int argc, char **argv)
{
  signal (SIGINT, intHandler);

  if (argc != 2)
    {
      fprintf (stderr, "usage: %s [wav file]\n", argv[0]);
      return 1;
}

  FILE *wavin = fopen (argv[1], "rb");
  if (wavin == NULL)
    {
      perror (argv[1]);
      return 1;
    }

  /* begin checking wav file the normative way */
  char RIFFTag[5] = { 0 };
  fread (RIFFTag, 1, 4, wavin);
  if (strcmp ("RIFF", RIFFTag))
    {
      fprintf (stderr, "Unrecognize file format: %s\n", argv[1]);
      return 1;
    }

  /* begin check sample rate */
  uint32_t samplerate;
  fseek (wavin, 24, SEEK_SET);
  fread (&samplerate, sizeof (uint32_t), 1, wavin);
  if (samplerate != 44100)
    {
      fprintf (stderr, "%s samplerate must be 44100 not %d\n", argv[1],
               samplerate);
      fclose (wavin);
      return 1;
    }
  /* end check samplerate */

  uint16_t audioformat;
  fseek (wavin, 20, SEEK_SET);
  fread (&audioformat, sizeof (uint16_t), 1, wavin);
  if (audioformat != 1)
    {
      fprintf (stderr, "%s must be a PCM!\n", argv[1]);
      fclose (wavin);
      return 1;
    }

  uint16_t channels;
  fseek (wavin, 22, SEEK_SET);
  fread (&channels, sizeof (uint16_t), 1, wavin);
  if (channels != 1)
    {
      fprintf (stderr, "%s must be a mono wav file!\n", argv[1]);
      fclose (wavin);
      return 1;
    }

  uint16_t blockalign;
  fseek (wavin, 32, SEEK_SET);
  fread (&blockalign, sizeof (uint16_t), 1, wavin);
  if (blockalign != 2)
    {
      fprintf (stderr, "%s must be a 16-bit wav file!\n", argv[1]);
      fclose (wavin);
      return 1;
    }

  char data_tag[4] = { 0 };
  fseek (wavin, 36, SEEK_SET);
  do
    {
      if (fread (data_tag, 1, 4, wavin) != 4)
        {
          fprintf (stderr, "No data tag!!\n");
          return 1;
        }
      if (!strncmp (data_tag, "data", 4))
        {
          break;
        }
      else
        fseek (wavin, -3, SEEK_CUR);
    }
  while (1);
/* begin count numsamples */
  uint32_t numsamples;
  fread (&numsamples, sizeof (uint32_t), 1, wavin);
  numsamples >>= 1;
  printf ("numsamples: %d\n", numsamples);
  /* end count numsamples */

  pcmin = malloc (sizeof (int16_t) * numsamples);
  pcmout = malloc (sizeof (int16_t) * numsamples);
  fread (pcmin, sizeof (int16_t), numsamples, wavin);
  fclose(wavin);

  const double delta_x = 1.0 / 44100.0, c = 1900.0, h = 1.0;

  v = malloc (sizeof (double *) * height);
  u = malloc (sizeof (double *) * height);
  assert (v != NULL && u != NULL);

#pragma omp parallel for
  for (int y = 0; y < height; y++)
    {
      v[y] = malloc (sizeof (double) * width);
      u[y] = malloc (sizeof (double) * width);
      assert (u[y] != NULL && v[y] != NULL);
    }

#pragma omp parallel for collapse(2)
  for (int x = 0; x < width; x++)
    for (int y = 0; y < height; y++)
      {
        u[y][x] = 0.0;          //init2d (x, y);
        v[y][x] = 0.0;
      }

  png_frame = malloc (sizeof (png_byte *) * height);
  assert (png_frame != NULL);

#pragma omp parallel for
  for (int y = 0; y < height; y++)
    assert ((png_frame[y] = malloc (4 * width)) != NULL);

/* This is where the simulation begins */
#pragma omp parallel for ordered schedule(dynamic)
  for (uint32_t file = 0; file < (numsamples * 25) / 44100; file++)
    {
      char buff[50] = { 0 };
#pragma omp ordered
      {
        sprintf (buff, "tt-%03d.png", file);

        printf ("\rWriting png frame: %s %d/%d... ", buff, file, ((numsamples * 25) / 44100) - 1);
        fflush (stdout);
        for (int i = 0; i < (int) (1.0 / (FPS * delta_x)); i++)
          {
            //int bb = printf ("%d/%d", i, (int) (1.0 / (FPS * delta_x)) - 1);
            //fflush (stdout);
            u[height / 2][width / 2] += (double) pcmin[pcm_ctr];
            for (int x = 0; x < width; x++)
              for (int y = 0; y < height; y++)
                {
                  v[y][x] +=
                    delta_x * c * c * (u[((y - 1) < 0) ? 0 : y - 1][x] +
                                       u[((y + 1) ==
                                          height) ? height - 1 : y + 1][x] +
                                       u[y][((x - 1) <
                                             0) ? 0 : x - 1] + u[y][((x +
                                                                      1) ==
                                                                     width) ?
                                                                    width -
                                                                    1 : x +
                                                                    1] -
                                       (4.0 * u[y][x])) / (h * h);
                }

            for (int x = 0; x < width; x++)
              for (int y = 0; y < height; y++)
                {
                  u[y][x] += delta_x * v[y][x];
                  u[y][x] *= 0.999;      // damp
                }
            pcmout[pcm_ctr] = (int16_t) u[(height/2) + 50][width / 2];
            pcm_ctr++;
            //for (int i = 0; i < bb; i++)
              //putchar ('\b');

          }
#pragma omp parallel for collapse(2)
        for (int x = 0; x < width; x++)
          for (int y = 0; y < height; y++)
            write_heatmap (u[y][x], &png_frame[y][x * 4]);
      }
      png_frame[(height/2) + 50 ][(4*width/2)] = 255;
      png_frame[(height/2) + 50 ][(4*width/2) + 1] = 0;
      png_frame[(height/2) + 50 ][(4*width/2) + 2] = 0;

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

      write_wav ();
      png_write_image (png, png_frame);
      png_write_end (png, NULL);
      fclose (fpout);
      png_destroy_write_struct (&png, &info);

    };
  free (pcmin);
  free (pcmout);

  if (u != NULL)
    for (int y = 0; y < height; y++)
      free (u[y]);

  if (v != NULL)
    for (int y = 0; y < height; y++)
      free (v[y]);

  if (png_frame != NULL)
    for (int y = 0; y < height; y++)
      free (png_frame[y]);

  free (png_frame);
  free (u);
  free (v);

  puts ("done");
  fflush (stdout);
  return 0;

}