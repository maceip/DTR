import {name} from '@gdp-ts/core';
import {PermissionPolicy, type AuthorizationRequest} from '../src/permissions.js';

const policy = new PermissionPolicy({version: 1, rules: []});
const request: AuthorizationRequest = {operation: 'read', arguments: {}, access: [{resource: 'notes:alice', action: 'read'}]};
name('alice', request, (subject, first) => {
  const proof = policy.authorize(subject, first);
  policy.consume(subject, first, proof);
  // @ts-expect-error a checked proof is mandatory
  policy.consume(subject, first);
  // @ts-expect-error JSON cannot construct a compiler proof
  policy.consume(subject, first, {kind: 'DtrAuthorized'});
  name(request, second => {
    // @ts-expect-error approval for one named request cannot authorize another
    policy.consume(subject, second, proof);
  });
  name('bob', other => {
    // @ts-expect-error approval is also bound to the authenticated subject
    policy.consume(other, first, proof);
  });
});
