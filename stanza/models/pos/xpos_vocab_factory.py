# This is the XPOS factory method generated automatically from stanza.models.pos.build_xpos_vocab_factory.
# Please don't edit it!

import logging

from stanza.models.pos.vocab import WordVocab, XPOSVocab
from stanza.models.pos.xpos_vocab_utils import XPOSDescription, XPOSType, build_xpos_vocab, choose_simplest_factory

# using a sublogger makes it easier to test in the unittests
logger = logging.getLogger('stanza.models.pos.xpos_vocab_factory')

XPOS_DESCRIPTIONS = {
    'af_afribooms'   : XPOSDescription(XPOSType.XPOS, ''),
    'ar_padt'        : XPOSDescription(XPOSType.XPOS, ''),
    'bg_btb'         : XPOSDescription(XPOSType.XPOS, ''),
    'ca_ancora'      : XPOSDescription(XPOSType.XPOS, ''),
    'cs_cac'         : XPOSDescription(XPOSType.XPOS, ''),
    'cs_cltt'        : XPOSDescription(XPOSType.XPOS, ''),
    'cs_fictree'     : XPOSDescription(XPOSType.XPOS, ''),
    'cs_pdt'         : XPOSDescription(XPOSType.XPOS, ''),
    'en_partut'      : XPOSDescription(XPOSType.XPOS, ''),
    'es_ancora'      : XPOSDescription(XPOSType.XPOS, ''),
    'es_combined'    : XPOSDescription(XPOSType.XPOS, ''),
    'fr_partut'      : XPOSDescription(XPOSType.XPOS, ''),
    'gd_arcosg'      : XPOSDescription(XPOSType.XPOS, ''),
    'gl_ctg'         : XPOSDescription(XPOSType.XPOS, ''),
    'gl_treegal'     : XPOSDescription(XPOSType.XPOS, ''),
    'grc_perseus'    : XPOSDescription(XPOSType.XPOS, ''),
    'hr_set'         : XPOSDescription(XPOSType.XPOS, ''),
    'is_gc'          : XPOSDescription(XPOSType.XPOS, ''),
    'is_icepahc'     : XPOSDescription(XPOSType.XPOS, ''),
    'is_modern'      : XPOSDescription(XPOSType.XPOS, ''),
    'it_combined'    : XPOSDescription(XPOSType.XPOS, ''),
    'it_isdt'        : XPOSDescription(XPOSType.XPOS, ''),
    'it_markit'      : XPOSDescription(XPOSType.XPOS, ''),
    'it_parlamint'   : XPOSDescription(XPOSType.XPOS, ''),
    'it_partut'      : XPOSDescription(XPOSType.XPOS, ''),
    'it_postwita'    : XPOSDescription(XPOSType.XPOS, ''),
    'it_twittiro'    : XPOSDescription(XPOSType.XPOS, ''),
    'it_vit'         : XPOSDescription(XPOSType.XPOS, ''),
    'la_perseus'     : XPOSDescription(XPOSType.XPOS, ''),
    'la_udante'      : XPOSDescription(XPOSType.XPOS, ''),
    'lt_alksnis'     : XPOSDescription(XPOSType.XPOS, ''),
    'lv_lvtb'        : XPOSDescription(XPOSType.XPOS, ''),
    'ro_nonstandard' : XPOSDescription(XPOSType.XPOS, ''),
    'ro_rrt'         : XPOSDescription(XPOSType.XPOS, ''),
    'ro_simonero'    : XPOSDescription(XPOSType.XPOS, ''),
    'sk_snk'         : XPOSDescription(XPOSType.XPOS, ''),
    'sl_ssj'         : XPOSDescription(XPOSType.XPOS, ''),
    'sl_sst'         : XPOSDescription(XPOSType.XPOS, ''),
    'sr_set'         : XPOSDescription(XPOSType.XPOS, ''),
    'ta_ttb'         : XPOSDescription(XPOSType.XPOS, ''),
    'uk_iu'          : XPOSDescription(XPOSType.XPOS, ''),

    'be_hse'         : XPOSDescription(XPOSType.WORD, None),
    'bxr_bdt'        : XPOSDescription(XPOSType.WORD, None),
    'cop_scriptorium': XPOSDescription(XPOSType.WORD, None),
    'cu_proiel'      : XPOSDescription(XPOSType.WORD, None),
    'cy_ccg'         : XPOSDescription(XPOSType.WORD, None),
    'da_ddt'         : XPOSDescription(XPOSType.WORD, None),
    'de_gsd'         : XPOSDescription(XPOSType.WORD, None),
    'de_hdt'         : XPOSDescription(XPOSType.WORD, None),
    'el_gdt'         : XPOSDescription(XPOSType.WORD, None),
    'el_gud'         : XPOSDescription(XPOSType.WORD, None),
    'en_atis'        : XPOSDescription(XPOSType.WORD, None),
    'en_combined'    : XPOSDescription(XPOSType.WORD, None),
    'en_craft'       : XPOSDescription(XPOSType.WORD, None),
    'en_eslspok'     : XPOSDescription(XPOSType.WORD, None),
    'en_ewt'         : XPOSDescription(XPOSType.WORD, None),
    'en_genia'       : XPOSDescription(XPOSType.WORD, None),
    'en_gum'         : XPOSDescription(XPOSType.WORD, None),
    'en_gumreddit'   : XPOSDescription(XPOSType.WORD, None),
    'en_mimic'       : XPOSDescription(XPOSType.WORD, None),
    'en_test'        : XPOSDescription(XPOSType.WORD, None),
    'es_gsd'         : XPOSDescription(XPOSType.WORD, None),
    'et_edt'         : XPOSDescription(XPOSType.WORD, None),
    'et_ewt'         : XPOSDescription(XPOSType.WORD, None),
    'eu_bdt'         : XPOSDescription(XPOSType.WORD, None),
    'fa_perdt'       : XPOSDescription(XPOSType.WORD, None),
    'fa_seraji'      : XPOSDescription(XPOSType.WORD, None),
    'fi_tdt'         : XPOSDescription(XPOSType.WORD, None),
    'fr_combined'    : XPOSDescription(XPOSType.WORD, None),
    'fr_gsd'         : XPOSDescription(XPOSType.WORD, None),
    'fr_parisstories': XPOSDescription(XPOSType.WORD, None),
    'fr_rhapsodie'   : XPOSDescription(XPOSType.WORD, None),
    'fr_sequoia'     : XPOSDescription(XPOSType.WORD, None),
    'fro_profiterole': XPOSDescription(XPOSType.WORD, None),
    'ga_idt'         : XPOSDescription(XPOSType.WORD, None),
    'ga_twittirish'  : XPOSDescription(XPOSType.WORD, None),
    'got_proiel'     : XPOSDescription(XPOSType.WORD, None),
    'grc_proiel'     : XPOSDescription(XPOSType.WORD, None),
    'grc_ptnk'       : XPOSDescription(XPOSType.WORD, None),
    'gv_cadhan'      : XPOSDescription(XPOSType.WORD, None),
    'hbo_ptnk'       : XPOSDescription(XPOSType.WORD, None),
    'he_combined'    : XPOSDescription(XPOSType.WORD, None),
    'he_htb'         : XPOSDescription(XPOSType.WORD, None),
    'he_iahltknesset': XPOSDescription(XPOSType.WORD, None),
    'he_iahltwiki'   : XPOSDescription(XPOSType.WORD, None),
    'hi_hdtb'        : XPOSDescription(XPOSType.WORD, None),
    'hsb_ufal'       : XPOSDescription(XPOSType.WORD, None),
    'hu_szeged'      : XPOSDescription(XPOSType.WORD, None),
    'hy_armtdp'      : XPOSDescription(XPOSType.WORD, None),
    'hy_bsut'        : XPOSDescription(XPOSType.WORD, None),
    'hyw_armtdp'     : XPOSDescription(XPOSType.WORD, None),
    'id_csui'        : XPOSDescription(XPOSType.WORD, None),
    'it_old'         : XPOSDescription(XPOSType.WORD, None),
    'ka_glc'         : XPOSDescription(XPOSType.WORD, None),
    'kk_ktb'         : XPOSDescription(XPOSType.WORD, None),
    'kmr_mg'         : XPOSDescription(XPOSType.WORD, None),
    'kpv_lattice'    : XPOSDescription(XPOSType.WORD, None),
    'ky_ktmu'        : XPOSDescription(XPOSType.WORD, None),
    'la_proiel'      : XPOSDescription(XPOSType.WORD, None),
    'lij_glt'        : XPOSDescription(XPOSType.WORD, None),
    'lt_hse'         : XPOSDescription(XPOSType.WORD, None),
    'lzh_kyoto'      : XPOSDescription(XPOSType.WORD, None),
    'mr_ufal'        : XPOSDescription(XPOSType.WORD, None),
    'mt_mudt'        : XPOSDescription(XPOSType.WORD, None),
    'myv_jr'         : XPOSDescription(XPOSType.WORD, None),
    'nb_bokmaal'     : XPOSDescription(XPOSType.WORD, None),
    'nds_lsdc'       : XPOSDescription(XPOSType.WORD, None),
    'nn_nynorsk'     : XPOSDescription(XPOSType.WORD, None),
    'nn_nynorsklia'  : XPOSDescription(XPOSType.WORD, None),
    'no_bokmaal'     : XPOSDescription(XPOSType.WORD, None),
    'orv_birchbark'  : XPOSDescription(XPOSType.WORD, None),
    'orv_rnc'        : XPOSDescription(XPOSType.WORD, None),
    'orv_torot'      : XPOSDescription(XPOSType.WORD, None),
    'ota_boun'       : XPOSDescription(XPOSType.WORD, None),
    'pcm_nsc'        : XPOSDescription(XPOSType.WORD, None),
    'pt_bosque'      : XPOSDescription(XPOSType.WORD, None),
    'pt_cintil'      : XPOSDescription(XPOSType.WORD, None),
    'pt_dantestocks' : XPOSDescription(XPOSType.WORD, None),
    'pt_gsd'         : XPOSDescription(XPOSType.WORD, None),
    'pt_petrogold'   : XPOSDescription(XPOSType.WORD, None),
    'pt_porttinari'  : XPOSDescription(XPOSType.WORD, None),
    'qpm_philotis'   : XPOSDescription(XPOSType.WORD, None),
    'qtd_sagt'       : XPOSDescription(XPOSType.WORD, None),
    'ru_gsd'         : XPOSDescription(XPOSType.WORD, None),
    'ru_poetry'      : XPOSDescription(XPOSType.WORD, None),
    'ru_syntagrus'   : XPOSDescription(XPOSType.WORD, None),
    'ru_taiga'       : XPOSDescription(XPOSType.WORD, None),
    'sa_vedic'       : XPOSDescription(XPOSType.WORD, None),
    'sme_giella'     : XPOSDescription(XPOSType.WORD, None),
    'swl_sslc'       : XPOSDescription(XPOSType.WORD, None),
    'sq_staf'        : XPOSDescription(XPOSType.WORD, None),
    'te_mtg'         : XPOSDescription(XPOSType.WORD, None),
    'tr_atis'        : XPOSDescription(XPOSType.WORD, None),
    'tr_boun'        : XPOSDescription(XPOSType.WORD, None),
    'tr_framenet'    : XPOSDescription(XPOSType.WORD, None),
    'tr_imst'        : XPOSDescription(XPOSType.WORD, None),
    'tr_kenet'       : XPOSDescription(XPOSType.WORD, None),
    'tr_penn'        : XPOSDescription(XPOSType.WORD, None),
    'tr_tourism'     : XPOSDescription(XPOSType.WORD, None),
    'ug_udt'         : XPOSDescription(XPOSType.WORD, None),
    'uk_parlamint'   : XPOSDescription(XPOSType.WORD, None),
    'vi_vtb'         : XPOSDescription(XPOSType.WORD, None),
    'wo_wtb'         : XPOSDescription(XPOSType.WORD, None),
    'xcl_caval'      : XPOSDescription(XPOSType.WORD, None),
    'zh-hans_gsdsimp': XPOSDescription(XPOSType.WORD, None),
    'zh-hant_gsd'    : XPOSDescription(XPOSType.WORD, None),
    'zh_gsdsimp'     : XPOSDescription(XPOSType.WORD, None),

    'en_lines'       : XPOSDescription(XPOSType.XPOS, '-'),
    'fo_farpahc'     : XPOSDescription(XPOSType.XPOS, '-'),
    'ja_gsd'         : XPOSDescription(XPOSType.XPOS, '-'),
    'ja_gsdluw'      : XPOSDescription(XPOSType.XPOS, '-'),
    'sv_lines'       : XPOSDescription(XPOSType.XPOS, '-'),
    'ur_udtb'        : XPOSDescription(XPOSType.XPOS, '-'),

    'fi_ftb'         : XPOSDescription(XPOSType.XPOS, ','),
    'orv_ruthenian'  : XPOSDescription(XPOSType.XPOS, ','),

    'id_gsd'         : XPOSDescription(XPOSType.XPOS, '+'),
    'ko_gsd'         : XPOSDescription(XPOSType.XPOS, '+'),
    'ko_kaist'       : XPOSDescription(XPOSType.XPOS, '+'),
    'ko_ksl'         : XPOSDescription(XPOSType.XPOS, '+'),
    'qaf_arabizi'    : XPOSDescription(XPOSType.XPOS, '+'),

    'la_ittb'        : XPOSDescription(XPOSType.XPOS, '|'),
    'la_llct'        : XPOSDescription(XPOSType.XPOS, '|'),
    'nl_alpino'      : XPOSDescription(XPOSType.XPOS, '|'),
    'nl_lassysmall'  : XPOSDescription(XPOSType.XPOS, '|'),
    'sv_talbanken'   : XPOSDescription(XPOSType.XPOS, '|'),

    'pl_lfg'         : XPOSDescription(XPOSType.XPOS, ':'),
    'pl_pdb'         : XPOSDescription(XPOSType.XPOS, ':'),
}

def xpos_vocab_factory(data, shorthand):
    if shorthand not in XPOS_DESCRIPTIONS:
        logger.warning("%s is not a known dataset.  Examining the data to choose which xpos vocab to use", shorthand)
    desc = choose_simplest_factory(data, shorthand)
    if shorthand in XPOS_DESCRIPTIONS:
        if XPOS_DESCRIPTIONS[shorthand] != desc:
            # log instead of throw
            # otherwise, updating datasets would be unpleasant
            logger.error("XPOS tagset in %s has apparently changed!  Was %s, is now %s", shorthand, XPOS_DESCRIPTIONS[shorthand], desc)
    else:
        logger.warning("Chose %s for the xpos factory for %s", desc, shorthand)
    return build_xpos_vocab(desc, data, shorthand)

