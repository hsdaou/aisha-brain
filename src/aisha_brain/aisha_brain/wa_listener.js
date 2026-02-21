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

    sock.ev.on('messages.upsert', ({ messages, type }) => {
        if (type !== 'notify') return;
        for (const msg of messages) {
            if (msg.key.fromMe) continue;           // ignore own messages
            if (!msg.message)   continue;           // ignore empty/status

            // Extract plain text — handle text, extended, and image captions
            const text =
                msg.message.conversation ||
                msg.message.extendedTextMessage?.text ||
                msg.message.imageMessage?.caption ||
                '';

            if (!text.trim()) continue;

            // Extract sender phone number (strip @s.whatsapp.net)
            const from = (msg.key.remoteJid || '').replace('@s.whatsapp.net', '');

            // Emit as JSON line for the Python node to read
            process.stdout.write(JSON.stringify({ from, message: text.trim() }) + '\n');
        }
    });
}

start().catch(err => {
    process.stderr.write(`Fatal: ${err}\n`);
    process.exit(1);
});
