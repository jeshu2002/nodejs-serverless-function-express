import { VercelRequest, VercelResponse } from '@vercel/node';
import fs from 'fs-extra';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  try {
    // Log to see if the function is being called
    console.log('Function is called.');

    // Read the models.json file asynchronously
    const modelsData = await fs.readFile('api/models.json', 'utf-8');
    
    // Log to see if reading the file is successful
    console.log('File read successfully.');

    const models = JSON.parse(modelsData);

    // Return the models data as JSON
    return res.json(models);
  } catch (error) {
    // Log the error to see what went wrong
    console.error('Error:', error);

    // Return an error response if reading the file fails
    return res.status(500).json({ error: 'Internal Server Error' });
  }
}
