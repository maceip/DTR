import parser from '@typescript-eslint/parser';
import gdp from '@gdp-ts/core/lint/eslint';

export default [
  {files: ['src/**/*.ts'], languageOptions: {parser}},
  ...gdp({files: ['src/permissions.ts', 'src/tool-gateway.ts', 'src/training.ts'],
    proofs: ['src/permissions.ts'], proofImports: ['permissions'], strict: true}),
];
