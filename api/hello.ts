import { VercelRequest, VercelResponse } from '@vercel/node';
import fs from 'fs-extra';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    // Read the models.json file asynchronously
    const modelsData = await fs.readFile('path/to/models.json', 'utf-8');
    const models = JSON.parse(modelsData);

    // Return the models data as JSON
    return res.json(models);
  } catch (error) {
    console.error('Error reading models.json:', error);
    // Return an error response if reading the file fails
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}
