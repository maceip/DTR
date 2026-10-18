import {PermissionPolicy} from './permissions.js';
import {ownedJson} from './owned-json.js';
import {RuntimeError, type FineTuneRequest, type Model, type ModelCapabilities} from './types.js';

export function modelCapabilities(model: string, definition: Model): ModelCapabilities {
  return {model, function_calling: {
    variants: definition.variants.map(variant => ({variant: variant.id, protocol: variant.chat?.tool_protocol || 'none', declared: !!variant.chat && variant.chat.tool_protocol !== 'none'})),
    execution: 'application_tool_gateway', quality_verified: false,
  }, fine_tuning: {available: false, status: 'not_implemented', objectives: ['function_calling', 'chat']}};
}

/** No dataset reader, network transport, trainer or fake job ID in this stub. */
export function fineTuneStub(policy: PermissionPolicy | undefined, subject: string, input: FineTuneRequest, definition: Model): never {
  const request = ownedJson(input, 'invalid_training_request');
  if (Object.keys(request).some(key => !['model', 'dataset', 'method', 'objective'].includes(key)) ||
      !request.model || !request.dataset || (request.method !== undefined && request.method !== 'lora') ||
      !['function_calling', 'chat'].includes(request.objective || 'function_calling')) throw new RuntimeError('invalid_training_request', 'Invalid fine-tuning contract');
  if (definition.task !== 'chat') throw new RuntimeError('wrong_operation', 'Fine-tuning requires a chat model');
  if (!policy) throw new RuntimeError('permission_denied', 'Fine-tuning requires explicit dataset permissions');
  return policy.run(subject, {operation: 'fine_tune', arguments: {...request}, access: [
    {resource: request.dataset, action: 'read'}, {resource: request.dataset, action: 'train'},
  ]}, () => { throw new RuntimeError('fine_tuning_unavailable', 'Fine-tuning is a contract stub; no trainer is installed. No data was read or uploaded.', {training_started: false}); });
}
