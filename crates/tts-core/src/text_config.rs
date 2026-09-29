/// Per-language text-normalization knobs, mirroring the `replace_characters`,
/// `remove_semicolons` and `pad_with_spaces_for_short_inputs` fields of the
/// official `Config` (`pocket_tts/utils/config.py`) and the six shipped
/// `<language>.yaml` files.
///
/// `append_terminal_punctuation` and `capitalize_first_letter` are not
/// represented here: every shipped language config leaves both at their
/// default of `true`, so `prepare_text_prompt` applies them unconditionally.
pub struct TextConfig {
    pub remove_semicolons: bool,
    pub replace_characters: &'static [(char, &'static str)],
    pub pad_with_spaces_for_short_inputs: bool,
}

impl TextConfig {
    pub const ENGLISH: TextConfig = TextConfig {
        remove_semicolons: false,
        replace_characters: &[],
        pad_with_spaces_for_short_inputs: false,
    };

    pub const FRENCH: TextConfig = TextConfig {
        remove_semicolons: true,
        replace_characters: &[
            ('"', ""),
            ('\u{201c}', ""), // “
            ('\u{201d}', ""), // ”
            ('\u{201e}', ""), // „
            ('\u{00ab}', ""), // «
            ('\u{00bb}', ""), // »
            ('\u{2019}', "'"), // ’
            ('\u{2018}', "'"), // ‘
            (':', ","),
            ('(', ""),
            (')', ""),
            ('[', ""),
            (']', ""),
        ],
        pad_with_spaces_for_short_inputs: false,
    };

    pub const GERMAN: TextConfig = TextConfig {
        remove_semicolons: true,
        replace_characters: &[
            ('"', ""),
            ('\u{201c}', ""),
            ('\u{201d}', ""),
            ('\u{201e}', ""),
            ('\u{00ab}', ""),
            ('\u{00bb}', ""),
            ('\u{2019}', "'"),
            ('\u{2018}', "'"),
            ('(', ""),
            (')', ""),
            ('[', ""),
            (']', ""),
        ],
        pad_with_spaces_for_short_inputs: false,
    };

    pub const SPANISH: TextConfig = TextConfig {
        remove_semicolons: false,
        replace_characters: &[
            ('"', ""),
            ('\u{201c}', ""),
            ('\u{201d}', ""),
            ('\u{201e}', ""),
            ('\u{00ab}', ""),
            ('\u{00bb}', ""),
            ('\u{2019}', "'"),
            ('\u{2018}', "'"),
            ('(', ""),
            (')', ""),
            ('[', ""),
            (']', ""),
            ('\u{00a1}', ""), // ¡
            ('\u{00bf}', ""), // ¿
        ],
        pad_with_spaces_for_short_inputs: false,
    };

    pub const PORTUGUESE: TextConfig = TextConfig {
        remove_semicolons: false,
        replace_characters: &[
            ('"', ""),
            ('\u{201c}', ""),
            ('\u{201d}', ""),
            ('\u{201e}', ""),
            ('\u{00ab}', ""),
            ('\u{00bb}', ""),
            ('\u{2019}', "'"),
            ('\u{2018}', "'"),
            ('(', ""),
            (')', ""),
            ('[', ""),
            (']', ""),
        ],
        pad_with_spaces_for_short_inputs: false,
    };

    pub const ITALIAN: TextConfig = TextConfig {
        remove_semicolons: false,
        replace_characters: &[
            ('"', ""),
            ('\u{201c}', ""),
            ('\u{201d}', ""),
            ('\u{201e}', ""),
            ('\u{00ab}', ""),
            ('\u{00bb}', ""),
            ('\u{2019}', "'"),
            ('\u{2018}', "'"),
            ('(', ""),
            (')', ""),
            ('[', ""),
            (']', ""),
        ],
        pad_with_spaces_for_short_inputs: false,
    };

    /// Looks up by the language identifiers used across this repo (full
    /// name or the short web-UI code), falling back to `ENGLISH` (no
    /// special-casing) for anything unrecognized.
    pub fn for_language(lang: &str) -> Self {
        match lang {
            "french" | "fr" => Self::FRENCH,
            "german" | "de" => Self::GERMAN,
            "spanish" | "es" => Self::SPANISH,
            "portuguese" | "pt" => Self::PORTUGUESE,
            "italian" | "it" => Self::ITALIAN,
            _ => Self::ENGLISH,
        }
    }
}
