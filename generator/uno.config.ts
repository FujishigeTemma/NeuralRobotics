import {
  defineConfig,
  presetIcons,
  presetTypography,
  presetUno,
  presetWebFonts,
  transformerVariantGroup,
} from "unocss";

const fonts = {
  mono: [
    {
      name: "Noto Sans Mono",
      weights: ["400", "500"],
    },
  ],
};

export default defineConfig({
  presets: [
    presetUno(),
    presetIcons(),
    presetTypography(),
    presetWebFonts({ provider: "none", fonts }),
  ],
  transformers: [transformerVariantGroup()],
});
