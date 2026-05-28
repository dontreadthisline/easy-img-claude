// @ts-check
import { defineConfig } from 'astro/config';
import starlight from '@astrojs/starlight';
import catppuccin from '@catppuccin/starlight';

// https://astro.build/config
export default defineConfig({
	outDir: './docs',
	base: '/easy-img-claude/',
	integrations: [
		starlight({
			title: 'img2text',
			logo: {
				src: './src/assets/logo.svg',
			},
			plugins: [catppuccin()],
			social: [{ icon: 'github', label: 'GitHub', href: 'https://github.com/dontreadthisline/easy-img-claude' }],
			sidebar: [
				{
					label: 'Guides',
					items: [
						{ label: 'Quick Start', slug: 'guides/quickstart' },
						{ label: 'Configuration', slug: 'guides/configuration' },
					],
				},
				{
					label: 'Reference',
					items: [{ label: 'CLI Reference', slug: 'reference/cli' }],
				},
			],
		}),
	],
});
