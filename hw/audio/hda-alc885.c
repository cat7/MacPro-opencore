/*
 * Realtek ALC885 HD Audio codec, as fitted to the Apple Mac Pro (1,1 / 2,1)
 *
 * Widget graph, capabilities and pin defaults of the Apple ALC885
 * (vendor 0x10ec0885, revision 0x100103, subsystem 0x106b0c00).
 *
 * SPDX-License-Identifier: GPL-2.0-or-later
 */

#include "qemu/osdep.h"
#include <math.h>
#include "hw/core/qdev-properties.h"
#include "intel-hda.h"
#include "intel-hda-defs.h"
#include "migration/vmstate.h"
#include "qapi/error.h"
#include "qapi/visitor.h"
#include "qemu/host-utils.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "qemu/audio.h"
#include "trace.h"
#include "qom/object.h"

#define ALC_NODES       0x27
#define ALC_MAX_CONN    11
#define ALC_COEFS       0x20
#define ALC_FRAMES      2048
#define ALC_TICK        SCALE_MS

/* widget types */
#define W_OUT       (AC_WID_AUD_OUT << AC_WCAP_TYPE_SHIFT)
#define W_IN        (AC_WID_AUD_IN << AC_WCAP_TYPE_SHIFT)
#define W_MIX       (AC_WID_AUD_MIX << AC_WCAP_TYPE_SHIFT)
#define W_PIN       (AC_WID_PIN << AC_WCAP_TYPE_SHIFT)
#define W_KNOB      (AC_WID_VOL_KNB << AC_WCAP_TYPE_SHIFT)
#define W_VENDOR    (AC_WID_VENDOR << AC_WCAP_TYPE_SHIFT)

#define PCM_ANALOG  0x000e0560  /* 16/20/24 bit, 44.1/48/96/192 kHz */
#define PCM_SPDIF_O 0x001e05e0  /* 16-32 bit, 44.1-192 kHz */
#define PCM_SPDIF_I 0x001e0560

#define AMP_ADC     0x80032e10  /* -16..+30 dB, 1 dB steps, mute */
#define AMP_LOOP    0x80051f17  /* -34.5..+12 dB, 1.5 dB steps, mute */
#define AMP_MUTE    0x80000000  /* mute only */
#define AMP_VOL     0x00034040  /* -64..0 dB, 1 dB steps */
#define AMP_BOOST   0x00270300  /* 0..30 dB, 10 dB steps */

typedef struct AlcNode {
    uint32_t wcaps;
    uint32_t pcm;
    uint32_t pincap;
    uint32_t amp_in;
    uint32_t amp_out;
    uint32_t config;
    uint8_t nconn;
    uint8_t conn[ALC_MAX_CONN];
} AlcNode;

#define DAC     { .wcaps = W_OUT | 0x11, .pcm = PCM_ANALOG }
#define ADC(m)  { .wcaps = W_IN | 0x11b, .pcm = PCM_ANALOG, \
                  .amp_in = AMP_ADC, .nconn = 1, .conn = { m } }
#define OMIX(d) { .wcaps = W_MIX | 0x10f, .amp_in = AMP_MUTE, \
                  .amp_out = AMP_VOL, .nconn = 2, .conn = { d, 0x0b } }
#define IMIX    { .wcaps = W_MIX | 0x10b, .amp_in = AMP_MUTE, .nconn = 11, \
                  .conn = { 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x14, 0x15, \
                            0x16, 0x17, 0x0b } }
#define PIN(cap, cfg) { .wcaps = W_PIN | 0x18f, .pincap = cap, \
                        .amp_in = AMP_BOOST, .amp_out = AMP_MUTE, \
                        .config = cfg, .nconn = 5, \
                        .conn = { 0x0c, 0x0d, 0x0e, 0x0f, 0x26 } }
#define VENDOR  { .wcaps = W_VENDOR }

/* pin defaults of the Mac Pro */
static const AlcNode alc885_nodes[ALC_NODES] = {
    [0x02] = DAC,
    [0x03] = DAC,
    [0x04] = DAC,
    [0x05] = DAC,
    [0x06] = { .wcaps = W_OUT | 0x211, .pcm = PCM_SPDIF_O },
    [0x07] = ADC(0x24),
    [0x08] = ADC(0x23),
    [0x09] = ADC(0x22),
    [0x0a] = { .wcaps = W_IN | 0x391, .pcm = PCM_SPDIF_I,
               .nconn = 1, .conn = { 0x1f } },
    [0x0b] = { .wcaps = W_MIX | 0x10b, .amp_in = AMP_LOOP,
               .nconn = 10, .conn = { 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
                                      0x14, 0x15, 0x16, 0x17 } },
    [0x0c] = OMIX(0x02),
    [0x0d] = OMIX(0x03),
    [0x0e] = OMIX(0x04),
    [0x0f] = OMIX(0x05),
    [0x10] = VENDOR,
    [0x11] = VENDOR,
    [0x12] = VENDOR,
    [0x13] = VENDOR,
    [0x14] = PIN(0x373c, 0x400000f0),
    [0x15] = PIN(0x373c, 0x01014150),   /* line out, rear */
    [0x16] = PIN(0x003c, 0x400000f0),
    [0x17] = PIN(0x003c, 0x400000f0),
    [0x18] = PIN(0x373c, 0x01217040),   /* headphone */
    [0x19] = PIN(0x373c, 0x01813110),   /* line in, rear */
    [0x1a] = PIN(0x373c, 0x90130130),   /* internal speaker */
    [0x1b] = PIN(0x373c, 0x400000f0),
    [0x1c] = { .wcaps = W_PIN | 0x001, .pincap = 0x20, .config = 0x400000f0 },
    [0x1d] = { .wcaps = W_PIN | 0x000, .pincap = 0x20, .config = 0x400000f0 },
    [0x1e] = { .wcaps = W_PIN | 0x300, .pincap = 0x10, .config = 0x0145e160,
               .nconn = 1, .conn = { 0x06 } },  /* optical out */
    [0x1f] = { .wcaps = W_PIN | 0x200, .pincap = 0x20,
               .config = 0x01c5e120 },          /* optical in */
    [0x20] = { .wcaps = W_VENDOR | 0x040 },
    [0x21] = { .wcaps = W_KNOB | 0x080 },
    [0x22] = IMIX,
    [0x23] = IMIX,
    [0x24] = IMIX,
    [0x25] = DAC,
    [0x26] = OMIX(0x25),
};

#define ALC_VENDOR_ID   0x10ec0885
#define ALC_REVISION_ID 0x00100103
#define ALC_SUBSYSTEM   0x106b0c00
#define ALC_AFG_CAP     0x00000d0d
#define ALC_GPIO_CAP    0x40000002  /* 2 GPIOs, unsolicited capable */
#define ALC_PROC_CAP    0x00001100  /* 17 coefficients */
#define ALC_KNOB_CAP    0x00000020  /* 32 steps */
#define ALC_PIN_LAST    0x1b
#define ALC_SPDIF_DAC   0x06
#define ALC_SPDIF_PIN   0x1e
#define ALC_LINE_IN     0x19
/* AppleHDA mutes the line-in mixer input while the jack senses empty */
#define ALC_JACKS_DEFAULT   (1u << (ALC_LINE_IN - 0x14))

#define TYPE_HDA_ALC885 "hda-alc885"
OBJECT_DECLARE_SIMPLE_TYPE(ALC885State, HDA_ALC885)

typedef struct ALC885Stream {
    ALC885State *s;
    bool output;
    uint8_t tag;
    bool open;
    bool running;
    uint32_t format;
    uint32_t frame;
    uint32_t bsize;
    struct audsettings as;
    union {
        SWVoiceIn *in;
        SWVoiceOut *out;
    } voice;
    char name[16];
    uint8_t *buf;
    int64_t rpos;
    int64_t wpos;
    int64_t start;
    QEMUTimer *timer;
} ALC885Stream;

struct ALC885State {
    HDACodecDevice hda;
    AudioBackend *audio_be;

    /* node state */
    uint8_t amp_in[ALC_NODES * ALC_MAX_CONN * 2];
    uint8_t amp_out[ALC_NODES * 2];
    uint8_t pinctl[ALC_NODES];
    uint8_t unsol[ALC_NODES];
    uint8_t connsel[ALC_NODES];
    uint8_t eapd[ALC_NODES];
    uint8_t sdi[ALC_NODES];
    uint8_t conv[ALC_NODES];
    uint16_t format[ALC_NODES];
    uint16_t digctl[ALC_NODES];
    uint32_t config[ALC_NODES];
    uint8_t power;
    uint8_t beep;
    uint8_t knob;
    uint8_t proc_state;
    uint16_t coef_index;
    uint16_t coef[ALC_COEFS];
    uint8_t gpio_data;
    uint8_t gpio_mask;
    uint8_t gpio_dir;
    uint8_t gpio_wake;
    uint8_t gpio_unsol;
    uint8_t gpio_sticky;
    uint32_t subsystem;
    bool running[2 * 16];

    ALC885Stream st[2][16];

    /* properties */
    uint32_t jacks;
    bool mixer;
};

static inline uint8_t *alc_amp_in(ALC885State *s, uint32_t nid,
                                  uint32_t idx, int ch)
{
    return &s->amp_in[(nid * ALC_MAX_CONN + idx) * 2 + ch];
}

static uint32_t alc_wtype(uint32_t nid)
{
    return (alc885_nodes[nid].wcaps & AC_WCAP_TYPE) >> AC_WCAP_TYPE_SHIFT;
}

static bool alc_is_conv(uint32_t nid)
{
    uint32_t t = alc_wtype(nid);

    return nid >= 2 && (t == AC_WID_AUD_OUT || t == AC_WID_AUD_IN);
}

/* -------------------------------------------------------------------------- */
/* signal path gains */

static double alc_amp_db(uint32_t caps, uint8_t val)
{
    uint32_t ofs = caps & 0x7f;
    uint32_t nsteps = (caps >> 8) & 0x7f;
    uint32_t step = (caps >> 16) & 0x7f;
    uint32_t gain = MIN(val & AC_AMP_GAIN, nsteps);

    return ((double)gain - (double)ofs) * (step + 1) * 0.25;
}

/* linear factor of an amp, 0 if muted */
static double alc_amp(uint32_t caps, uint8_t val)
{
    if ((caps & AC_AMPCAP_MUTE) && (val & AC_AMP_MUTE)) {
        return 0.0;
    }
    if (!(caps & 0x7f00)) {
        return 1.0;
    }
    return pow(10.0, alc_amp_db(caps, val) / 20.0);
}

/* gain from output converter @dac to the pins that are driven by it */
static void alc_out_gain(ALC885State *s, uint32_t dac, double g[2])
{
    uint32_t pin, m, ch;

    if (dac == ALC_SPDIF_DAC) {
        if ((s->digctl[dac] & AC_DIG1_ENABLE) &&
            !(s->digctl[dac] & AC_DIG1_NONAUDIO) &&
            (s->pinctl[ALC_SPDIF_PIN] & AC_PINCTL_OUT_EN)) {
            g[0] = MAX(g[0], 1.0);
            g[1] = MAX(g[1], 1.0);
        }
        return;
    }
    for (pin = 0x14; pin <= ALC_PIN_LAST; pin++) {
        if (!(s->pinctl[pin] & AC_PINCTL_OUT_EN)) {
            continue;
        }
        m = alc885_nodes[pin].conn[s->connsel[pin] % alc885_nodes[pin].nconn];
        if (alc885_nodes[m].conn[0] != dac) {
            continue;
        }
        for (ch = 0; ch < 2; ch++) {
            double v = alc_amp(alc885_nodes[m].amp_in,
                               *alc_amp_in(s, m, 0, ch))
                * alc_amp(alc885_nodes[m].amp_out, s->amp_out[m * 2 + ch])
                * alc_amp(alc885_nodes[pin].amp_out, s->amp_out[pin * 2 + ch]);
            g[ch] = MAX(g[ch], v);
        }
    }
}

/* gain from the pins to input converter @adc */
static void alc_in_gain(ALC885State *s, uint32_t adc, double g[2])
{
    uint32_t m, i, src, ch;

    if (alc885_nodes[adc].wcaps & AC_WCAP_DIGITAL) {
        return;
    }
    m = alc885_nodes[adc].conn[0];
    for (i = 0; i < alc885_nodes[m].nconn; i++) {
        src = alc885_nodes[m].conn[i];
        if (alc_wtype(src) != AC_WID_PIN ||
            !(s->pinctl[src] & AC_PINCTL_IN_EN)) {
            continue;
        }
        for (ch = 0; ch < 2; ch++) {
            double v = alc_amp(alc885_nodes[adc].amp_in,
                               *alc_amp_in(s, adc, 0, ch))
                * alc_amp(alc885_nodes[m].amp_in, *alc_amp_in(s, m, i, ch))
                * alc_amp(alc885_nodes[src].amp_in,
                          *alc_amp_in(s, src, 0, ch));
            g[ch] = MAX(g[ch], v);
        }
    }
}

/* -------------------------------------------------------------------------- */
/* streams */

static uint32_t alc_bytes_per_second(ALC885Stream *st)
{
    return st->frame * st->as.freq;
}

/*
 * The DMA side runs at the nominal sample rate of the virtual clock, as the
 * codec's own clock does; the ring absorbs the host side's jitter and drift.
 */
static int64_t alc_due(ALC885Stream *st, int64_t now, int64_t done)
{
    int64_t want;

    if (now <= st->start) {
        return 0;
    }
    want = muldiv64(now - st->start, alc_bytes_per_second(st),
                    NANOSECONDS_PER_SECOND);
    want -= want % st->frame;
    return MIN(want - done, (int64_t)st->bsize);
}

static void alc_output_timer(void *opaque)
{
    ALC885Stream *st = opaque;
    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    int64_t n = alc_due(st, now, st->wpos);
    int64_t over = (st->wpos - st->rpos) + n - st->bsize;

    if (over > 0) {
        /* host side is behind: drop the oldest frames */
        st->rpos += over;
        trace_hda_alc885_overrun(st->name);
    }
    while (n > 0) {
        uint32_t at = st->wpos % st->bsize;
        uint32_t chunk = MIN(st->bsize - at, n);

        if (!hda_codec_xfer(&st->s->hda, st->tag, true, st->buf + at, chunk)) {
            memset(st->buf + at, 0, chunk);
        }
        st->wpos += chunk;
        n -= chunk;
    }
    if (st->running) {
        timer_mod_anticipate_ns(st->timer, now + ALC_TICK);
    }
}

static void alc_output_cb(void *opaque, int avail)
{
    ALC885Stream *st = opaque;
    int64_t n = MIN(st->wpos - st->rpos, avail);

    while (n > 0) {
        uint32_t at = st->rpos % st->bsize;
        uint32_t chunk = MIN(st->bsize - at, n);
        size_t done = audio_be_write(st->s->audio_be, st->voice.out,
                                     st->buf + at, chunk);

        st->rpos += done;
        n -= done;
        if (done != chunk) {
            break;
        }
    }
}

static void alc_input_timer(void *opaque)
{
    ALC885Stream *st = opaque;
    int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);
    int64_t n = alc_due(st, now, st->rpos);
    int64_t under = n - (st->wpos - st->rpos);

    if (under > 0) {
        /* host side is behind: pad with silence */
        uint32_t at = st->wpos % st->bsize;
        uint32_t chunk = MIN(st->bsize - at, under);

        memset(st->buf + at, 0, chunk);
        memset(st->buf, 0, under - chunk);
        st->wpos += under;
    }
    while (n > 0) {
        uint32_t at = st->rpos % st->bsize;
        uint32_t chunk = MIN(st->bsize - at, n);

        hda_codec_xfer(&st->s->hda, st->tag, false, st->buf + at, chunk);
        st->rpos += chunk;
        n -= chunk;
    }
    if (st->running) {
        timer_mod_anticipate_ns(st->timer, now + ALC_TICK);
    }
}

static void alc_input_cb(void *opaque, int avail)
{
    ALC885Stream *st = opaque;
    int64_t n = MIN(st->bsize - (st->wpos - st->rpos), avail);

    while (n > 0) {
        uint32_t at = st->wpos % st->bsize;
        uint32_t chunk = MIN(st->bsize - at, n);
        size_t done = audio_be_read(st->s->audio_be, st->voice.in,
                                    st->buf + at, chunk);

        st->wpos += done;
        n -= done;
        if (done != chunk) {
            break;
        }
    }
}

static void alc_parse_format(uint32_t format, struct audsettings *as,
                             uint32_t *frame)
{
    static const uint8_t mult[8] = { 1, 2, 3, 4, 1, 1, 1, 1 };
    uint32_t bytes;

    as->freq = (format & AC_FMT_BASE_44K) ? 44100 : 48000;
    as->freq *= mult[(format & AC_FMT_MULT_MASK) >> AC_FMT_MULT_SHIFT];
    as->freq /= ((format & AC_FMT_DIV_MASK) >> AC_FMT_DIV_SHIFT) + 1;
    as->nchannels = ((format & AC_FMT_CHAN_MASK) >> AC_FMT_CHAN_SHIFT) + 1;
    as->big_endian = false;
    switch (format & AC_FMT_BITS_MASK) {
    case AC_FMT_BITS_8:
        as->fmt = AUDIO_FORMAT_S8;
        bytes = 1;
        break;
    case AC_FMT_BITS_16:
        as->fmt = AUDIO_FORMAT_S16;
        bytes = 2;
        break;
    default:
        /* 20 and 24 bit samples sit MSB-aligned in 32 bit containers */
        as->fmt = AUDIO_FORMAT_S32;
        bytes = 4;
        break;
    }
    *frame = bytes * as->nchannels;
}

static void alc_stream_set_running(ALC885Stream *st, bool running)
{
    if (st->running == running) {
        return;
    }
    st->running = running;
    trace_hda_alc885_running(st->name, running);
    if (running) {
        int64_t now = qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL);

        st->rpos = st->wpos = 0;
        st->start = now;
        timer_mod_anticipate_ns(st->timer, now + ALC_TICK);
    } else {
        timer_del(st->timer);
    }
    if (st->output) {
        audio_be_set_active_out(st->s->audio_be, st->voice.out, running);
    } else {
        audio_be_set_active_in(st->s->audio_be, st->voice.in, running);
    }
}

static void alc_stream_open(ALC885Stream *st, uint32_t format)
{
    ALC885State *s = st->s;

    alc_stream_set_running(st, false);
    st->format = format;
    alc_parse_format(format, &st->as, &st->frame);
    st->bsize = st->frame * ALC_FRAMES;
    g_free(st->buf);
    st->buf = g_malloc0(st->bsize);
    st->rpos = st->wpos = 0;
    trace_hda_alc885_format(st->name, st->as.nchannels, st->frame,
                            st->as.freq);
    if (st->output) {
        st->voice.out = audio_be_open_out(s->audio_be, st->voice.out,
                                          st->name, st, alc_output_cb,
                                          &st->as);
    } else {
        st->voice.in = audio_be_open_in(s->audio_be, st->voice.in,
                                        st->name, st, alc_input_cb,
                                        &st->as);
    }
    st->open = true;
}

static uint8_t alc_vol(double g)
{
    return (uint8_t)lrint(MIN(g, 1.0) * 255.0);
}

/* bind converters to stream tags, apply formats, gains and run state */
static void alc_update(ALC885State *s)
{
    uint32_t dir, tag, nid;

    for (dir = 0; dir < 2; dir++) {
        for (tag = 1; tag < 16; tag++) {
            ALC885Stream *st = &s->st[dir][tag];
            double g[2] = { 0.0, 0.0 };
            uint32_t first = 0;
            bool nonpcm = false;

            for (nid = 2; nid < ALC_NODES; nid++) {
                if (!alc_is_conv(nid) ||
                    (alc_wtype(nid) == AC_WID_AUD_OUT) != (dir == 1) ||
                    (s->conv[nid] >> 4) != tag) {
                    continue;
                }
                if (!first) {
                    first = nid;
                }
                if (s->format[nid] & AC_FMT_TYPE_NON_PCM) {
                    nonpcm = true;
                } else if (dir) {
                    alc_out_gain(s, nid, g);
                } else {
                    alc_in_gain(s, nid, g);
                }
            }
            if (!first) {
                if (st->open) {
                    alc_stream_set_running(st, false);
                }
                continue;
            }
            if (!st->open || st->format != s->format[first]) {
                alc_stream_open(st, s->format[first]);
            }
            if (nonpcm) {
                g[0] = g[1] = 0.0;
            }
            if (s->mixer) {
                bool mute = alc_vol(g[0]) == 0 && alc_vol(g[1]) == 0;

                trace_hda_alc885_gain(st->name, first, alc_vol(g[0]),
                                      alc_vol(g[1]));
                if (dir) {
                    audio_be_set_volume_out_lr(s->audio_be, st->voice.out,
                                               mute, alc_vol(g[0]),
                                               alc_vol(g[1]));
                } else {
                    audio_be_set_volume_in_lr(s->audio_be, st->voice.in,
                                              mute, alc_vol(g[0]),
                                              alc_vol(g[1]));
                }
            }
            alc_stream_set_running(st, s->running[dir * 16 + tag]);
        }
    }
}

/* -------------------------------------------------------------------------- */
/* jack sense */

static bool alc_present(ALC885State *s, uint32_t nid)
{
    return s->jacks & (1u << (nid - 0x14));
}

static void alc_jack_event(ALC885State *s, uint32_t old)
{
    uint32_t nid;

    for (nid = 0x14; nid <= 0x1f; nid++) {
        uint32_t bit = 1u << (nid - 0x14);

        if (!((old ^ s->jacks) & bit) || !(s->unsol[nid] & AC_UNSOL_ENABLED)) {
            continue;
        }
        trace_hda_alc885_unsol(nid, s->unsol[nid] & 0x3f,
                               !!(s->jacks & bit));
        hda_codec_response(&s->hda, false,
                           (uint32_t)(s->unsol[nid] & 0x3f) << 26);
    }
}

static void alc_get_jacks(Object *obj, Visitor *v, const char *name,
                          void *opaque, Error **errp)
{
    ALC885State *s = HDA_ALC885(obj);

    visit_type_uint32(v, name, &s->jacks, errp);
}

static void alc_set_jacks(Object *obj, Visitor *v, const char *name,
                          void *opaque, Error **errp)
{
    ALC885State *s = HDA_ALC885(obj);
    uint32_t val, old;

    if (!visit_type_uint32(v, name, &val, errp)) {
        return;
    }
    old = s->jacks;
    s->jacks = val & 0xfff;
    if (DEVICE(obj)->realized) {
        alc_jack_event(s, old);
    }
}

/* -------------------------------------------------------------------------- */
/* verbs */

static uint32_t alc_param(ALC885State *s, uint32_t nid, uint32_t id)
{
    const AlcNode *n = &alc885_nodes[nid];

    if (nid == 0) {
        switch (id) {
        case AC_PAR_VENDOR_ID:
            return ALC_VENDOR_ID;
        case AC_PAR_REV_ID:
            return ALC_REVISION_ID;
        case AC_PAR_NODE_COUNT:
            return 0x00010001;
        }
        return 0;
    }
    if (nid == 1) {
        switch (id) {
        case AC_PAR_NODE_COUNT:
            return (2 << 16) | (ALC_NODES - 2);
        case AC_PAR_FUNCTION_TYPE:
            return AC_FGT_UNSOL_CAP | AC_GRP_AUDIO_FUNCTION;
        case AC_PAR_AUDIO_FG_CAP:
            return ALC_AFG_CAP;
        case AC_PAR_PCM:
            return PCM_ANALOG;
        case AC_PAR_STREAM:
            return AC_SUPFMT_PCM;
        case AC_PAR_POWER_STATE:
            return 0x0000000f;
        case AC_PAR_GPIO_CAP:
            return ALC_GPIO_CAP;
        }
        return 0;
    }
    switch (id) {
    case AC_PAR_AUDIO_WIDGET_CAP:
        return n->wcaps;
    case AC_PAR_PCM:
        return n->pcm;
    case AC_PAR_STREAM:
        return n->pcm ? AC_SUPFMT_PCM : 0;
    case AC_PAR_PIN_CAP:
        return n->pincap;
    case AC_PAR_AMP_IN_CAP:
        return (n->wcaps & AC_WCAP_IN_AMP) ? n->amp_in : 0;
    case AC_PAR_AMP_OUT_CAP:
        return (n->wcaps & AC_WCAP_OUT_AMP) ? n->amp_out : 0;
    case AC_PAR_CONNLIST_LEN:
        return (n->wcaps & AC_WCAP_CONN_LIST) ? n->nconn : 0;
    case AC_PAR_PROC_CAP:
        return (n->wcaps & AC_WCAP_PROC_WID) ? ALC_PROC_CAP : 0;
    case AC_PAR_VOL_KNB_CAP:
        return alc_wtype(nid) == AC_WID_VOL_KNB ? ALC_KNOB_CAP : 0;
    }
    return 0;
}

static uint32_t alc_get_amp(ALC885State *s, uint32_t nid, uint32_t payload)
{
    int ch = (payload & AC_AMP_GET_LEFT) ? 0 : 1;
    uint32_t idx = payload & AC_AMP_GET_INDEX;

    if (payload & AC_AMP_GET_OUTPUT) {
        return s->amp_out[nid * 2 + ch];
    }
    if (idx >= ALC_MAX_CONN) {
        return 0;
    }
    return *alc_amp_in(s, nid, idx, ch);
}

static void alc_set_amp(ALC885State *s, uint32_t nid, uint32_t payload)
{
    uint8_t val = payload & (AC_AMP_MUTE | AC_AMP_GAIN);
    uint32_t idx = (payload & AC_AMP_SET_INDEX) >> AC_AMP_SET_INDEX_SHIFT;
    int ch;

    for (ch = 0; ch < 2; ch++) {
        if (!(payload & (ch ? AC_AMP_SET_RIGHT : AC_AMP_SET_LEFT))) {
            continue;
        }
        if (payload & AC_AMP_SET_OUTPUT) {
            s->amp_out[nid * 2 + ch] = val;
        }
        if ((payload & AC_AMP_SET_INPUT) && idx < ALC_MAX_CONN) {
            *alc_amp_in(s, nid, idx, ch) = val;
        }
    }
}

static void alc_reset_state(ALC885State *s)
{
    uint32_t nid, i;

    memset(s->amp_in, 0, sizeof(s->amp_in));
    memset(s->amp_out, 0, sizeof(s->amp_out));
    for (nid = 0; nid < ALC_NODES; nid++) {
        const AlcNode *n = &alc885_nodes[nid];

        for (i = 0; i < ALC_MAX_CONN; i++) {
            if (n->amp_in & AC_AMPCAP_MUTE) {
                *alc_amp_in(s, nid, i, 0) = AC_AMP_MUTE;
                *alc_amp_in(s, nid, i, 1) = AC_AMP_MUTE;
            }
        }
        if (n->amp_out & AC_AMPCAP_MUTE) {
            s->amp_out[nid * 2] = s->amp_out[nid * 2 + 1] = AC_AMP_MUTE;
        }
        s->pinctl[nid] = 0;
        s->unsol[nid] = 0;
        s->connsel[nid] = 0;
        s->eapd[nid] = 0;
        s->sdi[nid] = 0;
        s->conv[nid] = 0;
        s->format[nid] = AC_FMT_BITS_16 | (1 << AC_FMT_CHAN_SHIFT);
        s->digctl[nid] = 0;
    }
    s->power = 0;
    s->beep = 0;
    s->knob = 0x1f;
    s->proc_state = 0;
    s->coef_index = 0;
    memset(s->coef, 0, sizeof(s->coef));
    s->gpio_data = s->gpio_mask = s->gpio_dir = 0;
    s->gpio_wake = s->gpio_unsol = s->gpio_sticky = 0;
}

static void alc_cold_state(ALC885State *s)
{
    uint32_t nid;

    for (nid = 0; nid < ALC_NODES; nid++) {
        s->config[nid] = alc885_nodes[nid].config;
    }
    s->subsystem = ALC_SUBSYSTEM;
    alc_reset_state(s);
}

static uint32_t alc_verb(ALC885State *s, uint32_t nid, uint32_t verb,
                         uint32_t payload, bool *update)
{
    const AlcNode *n = &alc885_nodes[nid];
    uint32_t r, i, shift;

    switch (verb) {
    case AC_VERB_PARAMETERS:
        return alc_param(s, nid, payload);
    case AC_VERB_GET_SUBSYSTEM_ID:
        return s->subsystem;
    case 0x720 ... 0x723:
        shift = (verb - 0x720) * 8;
        s->subsystem = (s->subsystem & ~(0xffu << shift)) | (payload << shift);
        return 0;
    case AC_VERB_SET_CODEC_RESET:
        if (nid == 1) {
            alc_reset_state(s);
            *update = true;
        }
        return 0;
    case AC_VERB_GET_POWER_STATE:
        return (s->power << 4) | s->power;
    case AC_VERB_SET_POWER_STATE:
        s->power = payload & 0x0f;
        return 0;
    }

    if (nid == 1) {
        switch (verb) {
        case AC_VERB_GET_GPIO_DATA:
            return s->gpio_data & s->gpio_mask & s->gpio_dir;
        case AC_VERB_SET_GPIO_DATA:
            s->gpio_data = payload & 0x03;
            trace_hda_alc885_gpio(s->gpio_data, s->gpio_mask, s->gpio_dir);
            return 0;
        case AC_VERB_GET_GPIO_MASK:
            return s->gpio_mask;
        case AC_VERB_SET_GPIO_MASK:
            s->gpio_mask = payload & 0x03;
            trace_hda_alc885_gpio(s->gpio_data, s->gpio_mask, s->gpio_dir);
            return 0;
        case AC_VERB_GET_GPIO_DIRECTION:
            return s->gpio_dir;
        case AC_VERB_SET_GPIO_DIRECTION:
            s->gpio_dir = payload & 0x03;
            trace_hda_alc885_gpio(s->gpio_data, s->gpio_mask, s->gpio_dir);
            return 0;
        case AC_VERB_GET_GPIO_WAKE_MASK:
            return s->gpio_wake;
        case AC_VERB_SET_GPIO_WAKE_MASK:
            s->gpio_wake = payload & 0x03;
            return 0;
        case AC_VERB_GET_GPIO_UNSOLICITED_RSP_MASK:
            return s->gpio_unsol;
        case AC_VERB_SET_GPIO_UNSOLICITED_RSP_MASK:
            s->gpio_unsol = payload & 0x03;
            return 0;
        case AC_VERB_GET_GPIO_STICKY_MASK:
            return s->gpio_sticky;
        case AC_VERB_SET_GPIO_STICKY_MASK:
            s->gpio_sticky = payload & 0x03;
            return 0;
        case AC_VERB_GET_BEEP_CONTROL:
            return s->beep;
        case AC_VERB_SET_BEEP_CONTROL:
            s->beep = payload;
            return 0;
        }
        return 0;
    }
    if (nid == 0) {
        return 0;
    }

    switch (verb) {
    case AC_VERB_GET_CONNECT_SEL:
        return s->connsel[nid];
    case AC_VERB_SET_CONNECT_SEL:
        if (payload < n->nconn) {
            s->connsel[nid] = payload;
            *update = true;
        }
        return 0;
    case AC_VERB_GET_CONNECT_LIST:
        r = 0;
        for (i = 0; i < 4 && payload + i < n->nconn; i++) {
            r |= (uint32_t)n->conn[payload + i] << (i * 8);
        }
        return r;
    case AC_VERB_GET_AMP_GAIN_MUTE:
        return alc_get_amp(s, nid, payload);
    case AC_VERB_SET_AMP_GAIN_MUTE:
        alc_set_amp(s, nid, payload);
        *update = true;
        return 0;
    case AC_VERB_GET_STREAM_FORMAT:
        return alc_is_conv(nid) ? s->format[nid] : 0;
    case AC_VERB_SET_STREAM_FORMAT:
        if (alc_is_conv(nid)) {
            s->format[nid] = payload;
            *update = true;
        }
        return 0;
    case AC_VERB_GET_CONV:
        return alc_is_conv(nid) ? s->conv[nid] : 0;
    case AC_VERB_SET_CHANNEL_STREAMID:
        if (alc_is_conv(nid)) {
            s->conv[nid] = payload;
            *update = true;
        }
        return 0;
    case AC_VERB_GET_SDI_SELECT:
        return s->sdi[nid];
    case AC_VERB_SET_SDI_SELECT:
        s->sdi[nid] = payload & 0x0f;
        return 0;
    case AC_VERB_GET_PIN_WIDGET_CONTROL:
        return s->pinctl[nid];
    case AC_VERB_SET_PIN_WIDGET_CONTROL:
        if (alc_wtype(nid) == AC_WID_PIN) {
            s->pinctl[nid] = payload;
            *update = true;
        }
        return 0;
    case AC_VERB_GET_UNSOLICITED_RESPONSE:
        return s->unsol[nid];
    case AC_VERB_SET_UNSOLICITED_ENABLE:
        if (n->wcaps & AC_WCAP_UNSOL_CAP) {
            s->unsol[nid] = payload & (AC_UNSOL_ENABLED | 0x3f);
        }
        return 0;
    case AC_VERB_GET_PIN_SENSE:
        if (alc_wtype(nid) == AC_WID_PIN && (n->pincap & AC_PINCAP_PRES_DETECT)
            && alc_present(s, nid)) {
            return AC_PINSENSE_PRESENCE;
        }
        return 0;
    case AC_VERB_SET_PIN_SENSE:
        return 0;
    case AC_VERB_GET_EAPD_BTLENABLE:
        return s->eapd[nid];
    case AC_VERB_SET_EAPD_BTLENABLE:
        s->eapd[nid] = payload & 0x07;
        return 0;
    case AC_VERB_GET_DIGI_CONVERT_1:
        return s->digctl[nid];
    case AC_VERB_SET_DIGI_CONVERT_1:
        s->digctl[nid] = (s->digctl[nid] & 0xff00) | payload;
        *update = true;
        return 0;
    case AC_VERB_SET_DIGI_CONVERT_2:
        s->digctl[nid] = (s->digctl[nid] & 0x00ff) | ((payload & 0x7f) << 8);
        return 0;
    case AC_VERB_GET_VOLUME_KNOB_CONTROL:
        return s->knob;
    case AC_VERB_SET_VOLUME_KNOB_CONTROL:
        s->knob = payload;
        return 0;
    case AC_VERB_GET_CONFIG_DEFAULT:
        return alc_wtype(nid) == AC_WID_PIN ? s->config[nid] : 0;
    case AC_VERB_SET_CONFIG_DEFAULT_BYTES_0 ...
         AC_VERB_SET_CONFIG_DEFAULT_BYTES_3:
        if (alc_wtype(nid) == AC_WID_PIN) {
            shift = (verb - AC_VERB_SET_CONFIG_DEFAULT_BYTES_0) * 8;
            s->config[nid] = (s->config[nid] & ~(0xffu << shift)) |
                             (payload << shift);
        }
        return 0;
    case AC_VERB_GET_PROC_STATE:
        return s->proc_state;
    case AC_VERB_SET_PROC_STATE:
        s->proc_state = payload;
        return 0;
    case AC_VERB_GET_COEF_INDEX:
        return s->coef_index;
    case AC_VERB_SET_COEF_INDEX:
        s->coef_index = payload;
        return 0;
    case AC_VERB_GET_PROC_COEF:
        return s->coef_index < ALC_COEFS ? s->coef[s->coef_index] : 0;
    case AC_VERB_SET_PROC_COEF:
        if (s->coef_index < ALC_COEFS) {
            s->coef[s->coef_index] = payload;
        }
        return 0;
    }
    trace_hda_alc885_unhandled(nid, verb, payload);
    return 0;
}

static void alc_command(HDACodecDevice *hda, uint32_t nid, uint32_t data)
{
    ALC885State *s = HDA_ALC885(hda);
    uint32_t verb, payload, r = 0;
    bool update = false;

    if ((data & 0x70000) == 0x70000) {
        verb = (data >> 8) & 0xfff;
        payload = data & 0xff;
    } else {
        verb = (data >> 8) & 0xf00;
        payload = data & 0xffff;
    }
    if (nid < ALC_NODES) {
        r = alc_verb(s, nid, verb, payload, &update);
    }
    trace_hda_alc885_verb(nid, verb, payload, r);
    hda_codec_response(hda, true, r);
    if (update) {
        alc_update(s);
    }
}

static void alc_stream(HDACodecDevice *hda, uint32_t stnr, bool running,
                       bool output)
{
    ALC885State *s = HDA_ALC885(hda);

    if (stnr >= 16) {
        return;
    }
    s->running[output * 16 + stnr] = running;
    alc_update(s);
}

/* -------------------------------------------------------------------------- */

static void alc_realize(HDACodecDevice *hda, Error **errp)
{
    ALC885State *s = HDA_ALC885(hda);
    uint32_t dir, tag;

    if (!audio_be_check(&s->audio_be, errp)) {
        return;
    }
    for (dir = 0; dir < 2; dir++) {
        for (tag = 0; tag < 16; tag++) {
            ALC885Stream *st = &s->st[dir][tag];

            st->s = s;
            st->output = dir;
            st->tag = tag;
            snprintf(st->name, sizeof(st->name), "alc885.%s%u",
                     dir ? "out" : "in", tag);
            st->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,
                                     dir ? alc_output_timer : alc_input_timer,
                                     st);
        }
    }
    alc_cold_state(s);
}

static void alc_exit(HDACodecDevice *hda)
{
    ALC885State *s = HDA_ALC885(hda);
    uint32_t dir, tag;

    for (dir = 0; dir < 2; dir++) {
        for (tag = 0; tag < 16; tag++) {
            ALC885Stream *st = &s->st[dir][tag];

            timer_free(st->timer);
            if (st->open) {
                if (dir) {
                    audio_be_close_out(s->audio_be, st->voice.out);
                } else {
                    audio_be_close_in(s->audio_be, st->voice.in);
                }
            }
            g_free(st->buf);
        }
    }
}

static void alc_reset(DeviceState *dev)
{
    ALC885State *s = HDA_ALC885(dev);

    memset(s->running, 0, sizeof(s->running));
    alc_cold_state(s);
    alc_update(s);
}

static int alc_post_load(void *opaque, int version)
{
    ALC885State *s = opaque;
    uint32_t dir, tag;

    for (dir = 0; dir < 2; dir++) {
        for (tag = 0; tag < 16; tag++) {
            ALC885Stream *st = &s->st[dir][tag];

            alc_stream_set_running(st, false);
            st->format = ~0u;
        }
    }
    alc_update(s);
    return 0;
}

static const VMStateDescription vmstate_alc885 = {
    .name = "hda-alc885",
    .version_id = 1,
    .post_load = alc_post_load,
    .fields = (const VMStateField[]) {
        VMSTATE_UINT8_ARRAY(amp_in, ALC885State, ALC_NODES * ALC_MAX_CONN * 2),
        VMSTATE_UINT8_ARRAY(amp_out, ALC885State, ALC_NODES * 2),
        VMSTATE_UINT8_ARRAY(pinctl, ALC885State, ALC_NODES),
        VMSTATE_UINT8_ARRAY(unsol, ALC885State, ALC_NODES),
        VMSTATE_UINT8_ARRAY(connsel, ALC885State, ALC_NODES),
        VMSTATE_UINT8_ARRAY(eapd, ALC885State, ALC_NODES),
        VMSTATE_UINT8_ARRAY(sdi, ALC885State, ALC_NODES),
        VMSTATE_UINT8_ARRAY(conv, ALC885State, ALC_NODES),
        VMSTATE_UINT16_ARRAY(format, ALC885State, ALC_NODES),
        VMSTATE_UINT16_ARRAY(digctl, ALC885State, ALC_NODES),
        VMSTATE_UINT32_ARRAY(config, ALC885State, ALC_NODES),
        VMSTATE_UINT8(power, ALC885State),
        VMSTATE_UINT8(beep, ALC885State),
        VMSTATE_UINT8(knob, ALC885State),
        VMSTATE_UINT8(proc_state, ALC885State),
        VMSTATE_UINT16(coef_index, ALC885State),
        VMSTATE_UINT16_ARRAY(coef, ALC885State, ALC_COEFS),
        VMSTATE_UINT8(gpio_data, ALC885State),
        VMSTATE_UINT8(gpio_mask, ALC885State),
        VMSTATE_UINT8(gpio_dir, ALC885State),
        VMSTATE_UINT8(gpio_wake, ALC885State),
        VMSTATE_UINT8(gpio_unsol, ALC885State),
        VMSTATE_UINT8(gpio_sticky, ALC885State),
        VMSTATE_UINT32(subsystem, ALC885State),
        VMSTATE_UINT32(jacks, ALC885State),
        VMSTATE_BOOL_ARRAY(running, ALC885State, 2 * 16),
        VMSTATE_END_OF_LIST()
    }
};

static const Property alc_properties[] = {
    DEFINE_AUDIO_PROPERTIES(ALC885State, audio_be),
    DEFINE_PROP_BOOL("mixer", ALC885State, mixer, true),
};

static void alc_instance_init(Object *obj)
{
    HDA_ALC885(obj)->jacks = ALC_JACKS_DEFAULT;
    object_property_add(obj, "jacks", "uint32", alc_get_jacks, alc_set_jacks,
                        NULL, NULL);
    object_property_set_description(obj, "jacks",
                                    "Plugged jacks, bit n = pin node 0x14 + n"
                                    " (default 0x20: line in)");
}

static void alc_class_init(ObjectClass *klass, const void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);
    HDACodecDeviceClass *k = HDA_CODEC_DEVICE_CLASS(klass);

    k->init = alc_realize;
    k->exit = alc_exit;
    k->command = alc_command;
    k->stream = alc_stream;
    set_bit(DEVICE_CATEGORY_SOUND, dc->categories);
    device_class_set_legacy_reset(dc, alc_reset);
    dc->vmsd = &vmstate_alc885;
    dc->desc = "Realtek ALC885 HDA codec (Mac Pro)";
    device_class_set_props(dc, alc_properties);
}

static const TypeInfo alc_info = {
    .name          = TYPE_HDA_ALC885,
    .parent        = TYPE_HDA_CODEC_DEVICE,
    .instance_size = sizeof(ALC885State),
    .instance_init = alc_instance_init,
    .class_init    = alc_class_init,
};

static void alc_register_types(void)
{
    type_register_static(&alc_info);
}

type_init(alc_register_types)
