/**
 * wa_listener.js — WhatsApp incoming message listener for AI-SHA
 *
 * Reuses the mudslide auth credentials so no re-login is needed.
 * Prints received messages to stdout as JSON lines:
 *   {"from": "971XXXXXXXXX", "message": "hello"}
 *
 * Usage:
 *   node wa_listener.js
 *
 * Requires baileys (installed with mudslide):
 *   BAILEYS=/home/robot-wst/.npm/_npx/3d8e8e1f43a0b507/node_modules/baileys
 */

const path = require('path');
const fs   = require('fs');

// Silence all console.* output — only process.stdout.write (our JSON lines)
// and process.stderr.write (our status messages) should escape.
console.log   = () => {};
console.debug = () => {};
console.info  = () => {};
console.warn  = () => {};
console.error = () => {};

// ---- locate baileys next to mudslide ----
const BAILEYS_PATH = process.env.BAILEYS_PATH ||
    '/home/robot-wst/.npm/_npx/3d8e8e1f43a0b507/node_modules/baileys';

const {
    default: makeWASocket,
    useMultiFileAuthState,
    DisconnectReason,
    fetchLatestBaileysVersion,
} = require(BAILEYS_PATH);

// ---- mudslide auth folder ----
const AUTH_FOLDER = process.env.MUDSLIDE_CACHE_FOLDER ||
    path.join(process.env.HOME, '.local', 'share', 'mudslide');

async function start() {
    const { state, saveCreds } = await useMultiFileAuthState(AUTH_FOLDER);
    const { version } = await fetchLatestBaileysVersion();

    const sock = makeWASocket({
        version,
        auth: state,
        printQRInTerminal: false,   // already logged in via mudslide
        logger: require(path.join(BAILEYS_PATH, 'node_modules/pino'))({ level: 'silent' }),
        browser: ['AI-SHA', 'Chrome', '1.0.0'],
    });

    sock.ev.on('creds.update', saveCreds);

    sock.ev.on('connection.update', ({ connection, lastDisconnect }) => {
        if (connection === 'close') {
            const code = lastDisconnect?.error?.output?.statusCode;
            if (code === DisconnectReason.loggedOut) {
                process.stderr.write('ERROR: WhatsApp logged out. Run: npx mudslide login\n');
                process.exit(1);
            }
            // Any other disconnect — restart after 5s
            process.stderr.write(`Disconnected (code=${code}), reconnecting in 5s...\n`);
            setTimeout(start, 5000);
        } else if (connection === 'open') {
            process.stderr.write('WhatsApp connection open\n');
        }
    });

    // ---- Handle outgoing replies via stdin ----
    // Python sends JSON lines: {"to": "971XXXXXXXXX", "message": "..."}
    // If to === "me", send to the user's own JID (status@broadcast won't work;
    // use sock.user.id which Baileys sets after connection).
    const readline = require('readline');
    const rl = readline.createInterface({ input: process.stdin });
    rl.on('line', async (line) => {
        try {
            const { to, message } = JSON.parse(line.trim());
            if (!to || !message) return;

            let jid;
            if (to === 'me' || to === 'self') {
                // sock.user.id is our own JID set by Baileys after auth
                jid = sock.user?.id;
                if (!jid) {
                    process.stderr.write('SEND_ERR: no self JID yet\n');
                    return;
                }
            } else if (to.includes('@')) {
                // Full JID already provided (e.g. 269655394504744@lid or 971XXX@s.whatsapp.net)
                jid = to;
            } else {
                // Bare number — assume standard phone JID
                jid = `${to}@s.whatsapp.net`;
            }

            await sock.sendMessage(jid, { text: message });
            process.stderr.write(`SEND_OK: ${jid}\n`);
        } catch (err) {
            process.stderr.write(`SEND_ERR: ${err.message}\n`);
        }
    });

    sock.ev.on('messages.upsert', ({ messages, type }) => {
        if (type !== 'notify') return;
        for (const msg of messages) {
            if (!msg.message) continue;

            // Accept messages sent FROM the linked device (fromMe=true, i.e. the authorized user
            // messaging the robot from their own phone) OR received from others (fromMe=false,
            // i.e. a third party — filtered by allowed_number in whatsapp_listener.py).
            // Reject only protocol/app messages that have no remoteJid.
            const rawJid = msg.key.remoteJid || '';
            if (!rawJid) continue;

            // Extract plain text
            const text =
                msg.message.conversation ||
                msg.message.extendedTextMessage?.text ||
                msg.message.imageMessage?.caption ||
                msg.message.ephemeralMessage?.message?.conversation ||
                msg.message.viewOnceMessage?.message?.conversation ||
                '';

            if (!text.trim()) continue;

            // Keep both stripped sender (for auth matching) and full JID (for reply routing)
            const from = rawJid.replace(/@.+$/, '');
            const fromMe = !!msg.key.fromMe;

            process.stdout.write(JSON.stringify({ from, rawJid, message: text.trim(), fromMe }) + '\n');
        }
    });
}

start().catch(err => {
    process.stderr.write(`Fatal: ${err}\n`);
    process.exit(1);
});
