# Font assets / فایل‌های قلم

The original University of Tehran template expects font files in this directory
when the `irfonts` class option is enabled in `main.tex`. The font binaries
are intentionally not included in this Git-ready package and are ignored by Git.

To build with the original typography, copy the `font/*.ttf` files from **your
own original template archive** into this directory **locally**. Keep them
uncommitted. Check each font's license before publishing or sharing it.

With `irfonts` enabled, `tex/commands.tex` requires IRLotusICEE, IRTitr,
and LiberationSerif files by filename. Without `irfonts`, it uses system fonts
`XB Niloofar`, `XB Titre`, and `Junicode`, which must be installed separately.

برای ساخت پایان‌نامه با تنظیم فعلی، فایل‌های قلم موجود در نسخه اصلی قالب خودتان
را به‌صورت محلی داخل این پوشه قرار دهید و وارد مخزن Git نکنید.
