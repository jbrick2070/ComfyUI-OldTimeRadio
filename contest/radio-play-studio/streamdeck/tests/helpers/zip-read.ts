// SPDX-License-Identifier: MIT
// Reads a .streamDeckPlugin (a zip) without a library: the central directory
// gives each entry, its local header gives where the bytes start, and stored
// or deflated entries are the only two the Elgato packer writes.
import { crc32, inflateRawSync } from "node:zlib";

export interface ZipEntry {
  name: string;
  data: Buffer;
}

const END_OF_DIRECTORY = 0x06054b50;
const DIRECTORY_ENTRY = 0x02014b50;
const LOCAL_HEADER = 0x04034b50;

export function readZip(zip: Buffer): ZipEntry[] {
  let end = -1;
  for (let at = zip.length - 22; at >= Math.max(0, zip.length - 22 - 0xffff); at -= 1) {
    if (zip.readUInt32LE(at) === END_OF_DIRECTORY) {
      end = at;
      break;
    }
  }
  if (end < 0) {
    throw new Error("not a zip: no end of central directory");
  }
  const count = zip.readUInt16LE(end + 10);
  let at = zip.readUInt32LE(end + 16);
  const entries: ZipEntry[] = [];
  for (let i = 0; i < count; i += 1) {
    if (zip.readUInt32LE(at) !== DIRECTORY_ENTRY) {
      throw new Error(`bad central directory entry ${i}`);
    }
    const method = zip.readUInt16LE(at + 10);
    const crc = zip.readUInt32LE(at + 16);
    let packedSize = zip.readUInt32LE(at + 20);
    let size = zip.readUInt32LE(at + 24);
    const nameLength = zip.readUInt16LE(at + 28);
    const extraLength = zip.readUInt16LE(at + 30);
    const commentLength = zip.readUInt16LE(at + 32);
    let local = zip.readUInt32LE(at + 42);
    const name = zip.subarray(at + 46, at + 46 + nameLength).toString("utf8");
    // The Elgato packer writes zip64 entries: a field left at 0xFFFFFFFF is
    // carried, in this order, in the extra field with header id 1.
    const extra = zip.subarray(at + 46 + nameLength, at + 46 + nameLength + extraLength);
    for (let x = 0; x + 4 <= extra.length; ) {
      const id = extra.readUInt16LE(x);
      const length = extra.readUInt16LE(x + 2);
      if (id === 1) {
        let field = x + 4;
        const wide = (): number => {
          const value = Number(extra.readBigUInt64LE(field));
          field += 8;
          return value;
        };
        if (size === 0xffffffff) {
          size = wide();
        }
        if (packedSize === 0xffffffff) {
          packedSize = wide();
        }
        if (local === 0xffffffff) {
          local = wide();
        }
      }
      x += 4 + length;
    }
    at += 46 + nameLength + extraLength + commentLength;

    if (zip.readUInt32LE(local) !== LOCAL_HEADER) {
      throw new Error(`bad local header for ${name}`);
    }
    const start = local + 30 + zip.readUInt16LE(local + 26) + zip.readUInt16LE(local + 28);
    const packed = zip.subarray(start, start + packedSize);
    let data: Buffer;
    if (method === 0) {
      data = Buffer.from(packed);
    } else if (method === 8) {
      data = inflateRawSync(packed);
    } else {
      throw new Error(`${name}: compression method ${method} is not stored or deflate`);
    }
    if (data.length !== size || crc32(data) !== crc) {
      throw new Error(`${name}: size or checksum does not match its directory entry`);
    }
    entries.push({ name, data });
  }
  return entries;
}
