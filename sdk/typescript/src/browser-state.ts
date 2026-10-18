import {ChatResult, Message, RuntimeError, Session} from './types.js';
import {hash} from './browser-package.js';
import type {ToolActionStore, ToolResult} from './tool-gateway.js';

export function stable(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(stable).join(',') + ']';
  return '{' + Object.entries(value as Record<string, unknown>).filter(([, value]) => value !== undefined).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([key, value]) => JSON.stringify(key) + ':' + stable(value)).join(',') + '}';
}
export const fingerprint = (value: unknown): string => hash(new TextEncoder().encode(stable(value)));
const request = <T>(value: IDBRequest<T>): Promise<T> => new Promise((resolve, reject) => { value.onsuccess = () => resolve(value.result); value.onerror = () => reject(value.error); });
interface Turn { id: string; session: string; fingerprint: string; status: 'pending' | 'complete' | 'failed'; result?: ChatResult; error?: unknown; }

/** Canonical history survives page reloads. A crashed turn requires explicit recovery. */
export class BrowserState implements ToolActionStore {
  private database: Promise<IDBDatabase>;
  constructor(readonly name = 'phaseone-runtime-v1') {
    this.database = new Promise((resolve, reject) => {
      const open = indexedDB.open(name, 2);
      let blocked = false;
      open.onblocked = () => { blocked = true; reject(new RuntimeError('state_upgrade_blocked', 'Another tab is using an older runtime state format; close that tab before reopening', undefined, true)); };
      open.onupgradeneeded = () => {
        for (const store of ['sessions', 'turns', 'receipts', 'meta', 'actions']) if (!open.result.objectStoreNames.contains(store)) open.result.createObjectStore(store, {keyPath: 'id'});
      };
      open.onsuccess = () => { if (blocked) { open.result.close(); return; } open.result.onversionchange = () => open.result.close(); resolve(open.result); }; open.onerror = () => reject(open.error);
    });
  }
  private async transaction<T>(stores: string[], mode: IDBTransactionMode, callback: (tx: IDBTransaction) => Promise<T>): Promise<T> {
    const tx = (await this.database).transaction(stores, mode);
    const finished = new Promise<void>((resolve, reject) => { tx.oncomplete = () => resolve(); tx.onabort = () => reject(tx.error || new Error('Transaction aborted')); tx.onerror = () => reject(tx.error); });
    try { const result = await callback(tx); await finished; return result; }
    catch (error) { try { tx.abort(); } catch { /* Already committed or aborted. */ } await finished.catch(() => undefined); throw error; }
  }
  async create(packageId: string, model: string, messages: Message[]): Promise<Session> {
    const session = {id: crypto.randomUUID(), package: packageId, model, messages: structuredClone(messages), active_turn: null};
    await this.transaction(['sessions'], 'readwrite', async tx => { await request(tx.objectStore('sessions').add(session)); });
    return session;
  }
  session(id: string): Promise<Session> {
    return this.transaction(['sessions'], 'readonly', async tx => {
      const session = await request<Session | undefined>(tx.objectStore('sessions').get(id));
      if (!session) throw new RuntimeError('session_not_found', 'Browser session does not exist');
      return session;
    });
  }
  begin(id: string, turnId: string, value: unknown): Promise<{session: Session; cached?: ChatResult}> {
    if (!turnId || turnId.length > 200) return Promise.reject(new RuntimeError('invalid_turn_id', 'Turn ID must contain 1 to 200 characters'));
    const digest = fingerprint(value);
    return this.transaction(['sessions', 'turns'], 'readwrite', async tx => {
      const sessions = tx.objectStore('sessions'), turns = tx.objectStore('turns');
      const session = await request<Session | undefined>(sessions.get(id));
      if (!session) throw new RuntimeError('session_not_found', 'Browser session does not exist');
      const previous = await request<Turn | undefined>(turns.get(turnId));
      if (previous) {
        if (previous.session !== id || previous.fingerprint !== digest) throw new RuntimeError('idempotency_conflict', 'Turn ID was already used for different input');
        if (previous.status === 'complete') return {session, cached: {...previous.result!, cached: true}};
        throw new RuntimeError(previous.status === 'pending' ? 'turn_in_progress' : 'turn_failed', 'Inspect or recover the previous turn before starting another; it will not replay automatically', previous.error);
      }
      if (session.active_turn) throw new RuntimeError('session_busy', 'Session already has an active or interrupted turn', {turn_id: session.active_turn});
      session.active_turn = turnId;
      await request(sessions.put(session));
      await request(turns.add({id: turnId, session: id, fingerprint: digest, status: 'pending'} satisfies Turn));
      return {session};
    });
  }
  finish(sessionId: string, turnId: string, messages: Message[], result?: ChatResult, error?: unknown): Promise<void> {
    return this.transaction(['sessions', 'turns'], 'readwrite', async tx => {
      const sessions = tx.objectStore('sessions'), turns = tx.objectStore('turns');
      const session = await request<Session>(sessions.get(sessionId)), turn = await request<Turn>(turns.get(turnId));
      if (!session || !turn || session.active_turn !== turnId || turn.status !== 'pending') throw new RuntimeError('turn_state', 'Turn is no longer active');
      await request(sessions.put({...session, messages, active_turn: null}));
      await request(turns.put({...turn, status: error ? 'failed' : 'complete', result, error}));
    });
  }
  recover(sessionId: string): Promise<Session> {
    return this.transaction(['sessions', 'turns'], 'readwrite', async tx => {
      const sessions = tx.objectStore('sessions'), turns = tx.objectStore('turns');
      const session = await request<Session | undefined>(sessions.get(sessionId));
      if (!session) throw new RuntimeError('session_not_found', 'Browser session does not exist');
      if (session.active_turn) {
        const turn = await request<Turn>(turns.get(session.active_turn));
        await request(turns.put({...turn, status: 'failed', error: {code: 'interrupted_turn', message: 'Recovered explicitly; inspect paired-host actions before retrying'}}));
        session.active_turn = null; await request(sessions.put(session));
      }
      return session;
    });
  }
  append(kind: string, payload: unknown): Promise<Record<string, unknown>> {
    return this.transaction(['receipts', 'meta'], 'readwrite', async tx => {
      const meta = tx.objectStore('meta'), previous = await request<{id: string; sequence: number; hash: string} | undefined>(meta.get('audit'));
      const entry = {id: (previous?.sequence || 0) + 1, timestamp: new Date().toISOString(), kind, payload, previous_hash: previous?.hash || '0'.repeat(64)};
      const receipt = {...entry, hash: fingerprint(entry)};
      await request(tx.objectStore('receipts').add(receipt));
      await request(meta.put({id: 'audit', sequence: entry.id, hash: receipt.hash}));
      return receipt;
    });
  }
  receipts(): Promise<Record<string, unknown>[]> { return this.transaction(['receipts'], 'readonly', tx => request(tx.objectStore('receipts').getAll())); }
  beginAction(id: string, fingerprint: string): Promise<ToolResult | undefined> {
    return this.transaction(['actions'], 'readwrite', async tx => {
      const store = tx.objectStore('actions');
      const previous = await request<{id: string; fingerprint: string; result?: ToolResult} | undefined>(store.get(id));
      if (previous) {
        if (previous.fingerprint !== fingerprint) throw new RuntimeError('idempotency_conflict', 'Action ID was used for another request');
        if (!previous.result || previous.result.status === 'uncertain') throw new RuntimeError('action_outcome_unknown', 'Inspect the previous action; automatic replay is blocked');
        return previous.result;
      }
      await request(store.add({id, fingerprint}));
    });
  }
  finishAction(id: string, result: ToolResult): Promise<void> {
    return this.transaction(['actions'], 'readwrite', async tx => {
      const store = tx.objectStore('actions'), entry = await request<{id: string; fingerprint: string; result?: ToolResult} | undefined>(store.get(id));
      if (!entry || entry.result) throw new RuntimeError('action_conflict', 'Action is not pending');
      await request(store.put({...entry, result}));
    });
  }
  async close(): Promise<void> { (await this.database).close(); }
}
